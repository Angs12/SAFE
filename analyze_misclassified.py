#!/usr/bin/env python3
"""Deep-dive on misclassified inline pairs: compare ASM side-by-side."""

import sqlite3, os, sys, random, json
import torch
from tqdm import tqdm

from utils.evaluation import Evaluator
from safetorch.safe_network import SAFE
from utils.db import load_instructions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PAIRS = 50000
TOP_K = 10

DB_NORMAL = "small_test.db"
DB_INLINE = "small_test_inline.db"


def get_common_ids(db1, db2):
    conn1 = sqlite3.connect(db1)
    cur1 = conn1.cursor()
    conn2 = sqlite3.connect(db2)
    cur2 = conn2.cursor()

    cur1.execute("SELECT id, project, file_name, optimization, function_name FROM functions")
    map1 = {(r[1], r[2], r[3], r[4]): r[0] for r in tqdm(cur1, desc="Load DB1 keys")}
    cur2.execute("SELECT id, project, file_name, optimization, function_name FROM functions")
    map2 = {(r[1], r[2], r[3], r[4]): r[0] for r in tqdm(cur2, desc="Load DB2 keys")}

    common_keys = set(map1.keys()) & set(map2.keys())
    print(f"  {len(map1):,} vs {len(map2):,} functions, {len(common_keys):,} common", flush=True)

    ids1 = {map1[k] for k in common_keys}
    ids2 = {map2[k] for k in common_keys}

    conn1.close()
    conn2.close()
    return ids1, ids2, common_keys, map1, map2


def sample_pairs(db_path, valid_ids, max_pairs):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    valid_list = list(valid_ids)
    n_sample = max_pairs // 2

    def batched_query(label, limit=None):
        results = set()
        batch_size = 900
        for start in tqdm(range(0, len(valid_list), batch_size), desc=f"  Query label={label}", leave=False):
            batch = valid_list[start:start + batch_size]
            ph = ",".join("?" * len(batch))
            sql = "SELECT id1,id2 FROM pairs WHERE id1 IN ({}) AND id2 IN ({}) AND label={}".format(ph, ph, label)
            for row in cur.execute(sql, batch * 2):
                results.add((row[0], row[1]))
                if limit and len(results) >= limit:
                    return results
        return results

    true_set = batched_query(1, n_sample)
    print(f"  True pairs found: {len(true_set):,}", flush=True)
    true_sample = random.sample(list(true_set), min(len(true_set), n_sample))

    false_set = batched_query(0, n_sample)
    print(f"  False pairs found: {len(false_set):,}", flush=True)
    false_sample = list(false_set)[:min(len(false_set), len(true_sample))]

    conn.close()
    return true_sample, false_sample


def get_id2word():
    with open("model/word2id.json") as f:
        word2id = json.load(f)
    id2word = {v: k for k, v in word2id.items()}
    print(f"  Loaded vocabulary: {len(id2word):,} entries", flush=True)
    return id2word


def load_instr_text(db_path, fids, id2word):
    """Return {fid: [instruction_string, ...]} decoded from token IDs."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    result = {}
    batch = list(fids)
    for start in tqdm(range(0, len(batch), 500), desc=f"  Decode ASM from {os.path.basename(db_path)}", leave=False):
        chunk = batch[start:start + 500]
        ph = ",".join("?" * len(chunk))
        cur.execute(f"SELECT id, instructions_list FROM filtered_functions WHERE id IN ({ph})", chunk)
        for row in cur:
            tokens = json.loads(row[1])
            instrs = []
            for tid in tokens:
                word = id2word.get(tid - 1, "<UNK>")
                # Strip X_ prefix for readability
                if word.startswith("X_"):
                    word = word[2:]
                instrs.append(word)
            result[row[0]] = instrs
    conn.close()
    return result


def print_side_by_side(label, a_id_n, a_id_i, b_id_n, b_id_i, score,
                        key_a, key_b, asm_n, asm_i):
    _, fn_a, opt_a, fname_a = key_a
    _, fn_b, opt_b, fname_b = key_b

    instr_pairs = [
        ("A", a_id_n, a_id_i),
        ("B", b_id_n, b_id_i),
    ]

    for side, id_n, id_i in instr_pairs:
        instr_n = asm_n.get(id_n, [])
        instr_i = asm_i.get(id_i, [])
        max_len = max(len(instr_n), len(instr_i))
        prefix = os.path.basename(fn_a if side == "A" else fn_b)
        fname = fname_a if side == "A" else fname_b

        print(f"  ┌─ {side} ({prefix}, {fname}): normal={id_n} ({len(instr_n)} instrs)  inline={id_i} ({len(instr_i)} instrs)")
        print(f"  │  {'Normal':<50}│{'Inline':<50}")
        for idx in range(max_len):
            left = instr_n[idx] if idx < len(instr_n) else ""
            right = instr_i[idx] if idx < len(instr_i) else ""
            marker = "  │  "
            print(f"{marker}{left:<50}│{right:<50}")
        if len(instr_n) != len(instr_i):
            print(f"  │  {'─' * 50}┴{'─' * 50}")
            print(f"  │  Δ count: {len(instr_n)} → {len(instr_i)}")
        print()


def main():
    print(f"Device: {DEVICE}", flush=True)

    print(f"[1/6] Loading model & vocabulary...", flush=True)
    safe, _ = SAFE.load("model", DEVICE)
    id2word = get_id2word()

    print(f"[2/6] Finding common functions...", flush=True)
    ids1, ids2, common_keys, key_to_id_n, key_to_id_i = get_common_ids(DB_NORMAL, DB_INLINE)
    id_to_key_n = {v: k for k, v in key_to_id_n.items()}
    id_to_key_i = {v: k for k, v in key_to_id_i.items()}

    print(f"[3/6] Sampling pairs from inline DB...", flush=True)
    true_pairs, false_pairs = sample_pairs(DB_INLINE, ids2, MAX_PAIRS)

    pairs = [(a, b, 1) for a, b in true_pairs] + [(a, b, 0) for a, b in false_pairs]
    random.shuffle(pairs)
    fids = sorted(set(i for p in pairs for i in p[:2]))

    print(f"[4/6] Loading instructions for {len(fids):,} functions...", flush=True)
    instr_map = load_instructions(DB_INLINE, fids)

    print(f"[5/6] Embedding and scoring {len(pairs):,} pairs...", flush=True)
    evaluator = Evaluator(DEVICE)
    embeddings = evaluator.embed_all(safe, instr_map)
    scores, labels = evaluator.score_pairs(embeddings, pairs)

    print(f"[6/6] Analyzing misclassifications...", flush=True)
    false_positives = []
    false_negatives = []
    for (a, b, label), score in zip(pairs, scores):
        pred = 1 if score >= 0.5 else 0
        if pred != label:
            if label == 0:
                false_positives.append((score, a, b))
            else:
                false_negatives.append((score, a, b))

    false_positives.sort(key=lambda x: -x[0])
    false_negatives.sort(key=lambda x: x[0])

    print(f"  FP={len(false_positives):,}  FN={len(false_negatives):,}", flush=True)

    # Collect unique function IDs for ASM comparison
    normal_ids_needed = set()
    inline_ids_needed = set()
    misclassified = []
    for score, a_id, b_id in false_positives[:TOP_K]:
        key_a = id_to_key_i.get(a_id)
        key_b = id_to_key_i.get(b_id)
        if not key_a or not key_b:
            continue
        id_a_n = key_to_id_n.get(key_a)
        id_b_n = key_to_id_n.get(key_b)
        if not id_a_n or not id_b_n:
            continue
        normal_ids_needed.add(id_a_n)
        normal_ids_needed.add(id_b_n)
        inline_ids_needed.add(a_id)
        inline_ids_needed.add(b_id)
        misclassified.append(("FP", score, a_id, b_id, id_a_n, id_b_n, key_a, key_b))

    for score, a_id, b_id in false_negatives[:TOP_K]:
        key_a = id_to_key_i.get(a_id)
        key_b = id_to_key_i.get(b_id)
        if not key_a or not key_b:
            continue
        id_a_n = key_to_id_n.get(key_a)
        id_b_n = key_to_id_n.get(key_b)
        if not id_a_n or not id_b_n:
            continue
        normal_ids_needed.add(id_a_n)
        normal_ids_needed.add(id_b_n)
        inline_ids_needed.add(a_id)
        inline_ids_needed.add(b_id)
        misclassified.append(("FN", score, a_id, b_id, id_a_n, id_b_n, key_a, key_b))

    print(f"  Decoding ASM for {len(normal_ids_needed):,} normal + {len(inline_ids_needed):,} inline funcs...", flush=True)
    asm_normal = load_instr_text(DB_NORMAL, normal_ids_needed, id2word)
    asm_inline = load_instr_text(DB_INLINE, inline_ids_needed, id2word)

    # Print FP section
    fp_count = sum(1 for m in misclassified if m[0] == "FP")
    print(f"\n{'=' * 100}")
    print(f"FALSE POSITIVES ({fp_count} shown)")
    print(f"{'=' * 100}")
    for label, score, a_id, b_id, id_a_n, id_b_n, key_a, key_b in misclassified:
        if label != "FP":
            continue
        _, fn, opt, fname = key_a
        print(f"\n  FP score={score:.4f}  ({os.path.basename(fn)}, {fname})")
        print(f"  Inline pair: ({a_id}, {b_id})  Normal pair: ({id_a_n}, {id_b_n})")
        print_side_by_side("FP", id_a_n, a_id, id_b_n, b_id, score,
                           key_a, key_b, asm_normal, asm_inline)

    # Print FN section
    fn_count = sum(1 for m in misclassified if m[0] == "FN")
    print(f"\n{'=' * 100}")
    print(f"FALSE NEGATIVES ({fn_count} shown)")
    print(f"{'=' * 100}")
    for label, score, a_id, b_id, id_a_n, id_b_n, key_a, key_b in misclassified:
        if label != "FN":
            continue
        _, fn, opt, fname = key_a
        print(f"\n  FN score={score:.4f}  ({os.path.basename(fn)}, {fname})")
        print(f"  Inline pair: ({a_id}, {b_id})  Normal pair: ({id_a_n}, {id_b_n})")
        print_side_by_side("FN", id_a_n, a_id, id_b_n, b_id, score,
                           key_a, key_b, asm_normal, asm_inline)

    metrics = evaluator.compute_metrics(scores, labels)
    print(f"\n{'=' * 60}")
    print(f"Summary (inline DB only):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Threshold: {metrics['best_threshold']:.4f}")
    print(f"  FP={len(false_positives):,}  FN={len(false_negatives):,}  Total={len(pairs):,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
