#!/usr/bin/env python3
import sqlite3, json, os, sys, random
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from tqdm import tqdm
from safetorch.safe_network import SAFE
from safetorch.parameters import Config
from utils.function_normalizer import FunctionNormalizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_INSTRUCTIONS = 150


def load_model(model_dir):
    safe = SAFE(Config())
    safe.load_state_dict(
        torch.load(os.path.join(model_dir, "SAFEtorch.pt"), map_location=DEVICE)
    )
    return safe.to(DEVICE).eval(), FunctionNormalizer(MAX_INSTRUCTIONS)


def get_test_pairs(db_path, max_pairs=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pairs'")
    if cur.fetchone():
        cur.execute("SELECT id1,id2 FROM pairs WHERE label=1")
        true_pairs = [list(r) for r in cur.fetchall()]
        n_false = cur.execute("SELECT COUNT(*) FROM pairs WHERE label=0").fetchone()[0]
        want = max_pairs if max_pairs and n_false > max_pairs else n_false
        if want == n_false:
            cur.execute("SELECT id1,id2 FROM pairs WHERE label=0")
            false_pairs = [list(r) for r in cur.fetchall()]
        else:
            n_chunks = max(1, min(100, want // 1000))
            chunk_size = want // n_chunks
            seen = set()
            false_pairs = []
            for _ in range(n_chunks * 3):
                off = random.randint(0, max(0, n_false - chunk_size))
                cur.execute(
                    "SELECT id1,id2 FROM pairs WHERE label=0 LIMIT ? OFFSET ?",
                    (chunk_size, off),
                )
                for row in cur.fetchall():
                    key = (row[0], row[1])
                    if key not in seen:
                        seen.add(key)
                        false_pairs.append([row[0], row[1]])
                        if len(false_pairs) >= want:
                            break
                if len(false_pairs) >= want:
                    break
    else:
        r = cur.execute(
            "SELECT true_pair,false_pair FROM test_pairs WHERE id=0"
        ).fetchone()
        true_pairs = json.loads(r[0])
        false_pairs = json.loads(r[1])
    conn.close()
    return true_pairs, false_pairs


def load_instructions(db_path, function_ids):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    instr = {}
    for i in range(0, len(function_ids), 900):
        batch = function_ids[i : i + 900]
        cur.execute(
            f"SELECT id,instructions_list FROM filtered_functions "
            f"WHERE id IN ({','.join('?'*len(batch))})",
            batch,
        )
        for row in cur.fetchall():
            instr[row[0]] = json.loads(row[1])
    conn.close()
    return instr


def compute_embeddings(safe, normalizer, instr):
    embs = {}
    for fid, ids in tqdm(instr.items(), desc="Embeddings"):
        norm, lens = normalizer.normalize_functions([ids])
        with torch.no_grad():
            embs[fid] = safe(torch.LongTensor(norm[0]).to(DEVICE), torch.LongTensor(lens)).detach().cpu()
    return embs


def evaluate_pairs(embeddings, true_pairs, false_pairs):
    scores, labels = [], []
    for pair in tqdm(true_pairs, desc="True pairs"):
        scores.append(
            torch.cosine_similarity(embeddings[pair[0]], embeddings[pair[1]]).item()
        )
        labels.append(1)
    for pair in tqdm(false_pairs, desc="False pairs"):
        scores.append(
            torch.cosine_similarity(embeddings[pair[0]], embeddings[pair[1]]).item()
        )
        labels.append(0)
    return np.array(scores), np.array(labels)


def plot_metrics(scores, labels, output_dir):
    n = len(scores)
    thresh = np.linspace(0.01, 0.99, 200)

    print(f"  Computing metrics across {len(thresh)} thresholds...", flush=True)
    tp = np.zeros(200)
    fp = np.zeros(200)
    fn = np.zeros(200)
    tn = np.zeros(200)
    chunk = 200000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        cs = scores[start:end]
        cl = labels[start:end]
        preds = cs[None, :] >= thresh[:, None]
        for i in range(200):
            p = preds[i]
            tp[i] += np.sum(p & (cl == 1))
            fp[i] += np.sum(p & (cl == 0))
            fn[i] += np.sum((~p) & (cl == 1))
            tn[i] += np.sum((~p) & (cl == 0))

    total = tp + fp + fn + tn
    acc = np.divide(tp + tn, total, where=total > 0, out=np.zeros(200))
    prec = np.divide(tp, tp + fp, where=(tp + fp) > 0, out=np.zeros(200))
    rec = np.divide(tp, tp + fn, where=(tp + fn) > 0, out=np.zeros(200))
    pr_plus_rec = prec + rec
    f1 = np.divide(2 * prec * rec, pr_plus_rec, where=pr_plus_rec > 0, out=np.zeros(200))

    print("  Computing ROC / PR curves...", flush=True)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    prc, rec_curve, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    bi = np.argmax(f1)
    bt = thresh[bi]

    print("\n" + "=" * 45)
    print(f"  Best threshold:          {bt:.4f}")
    print(f"  Accuracy:                {acc[bi]:.4f}")
    print(f"  Precision:               {prec[bi]:.4f}")
    print(f"  Recall:                  {rec[bi]:.4f}")
    print(f"  F1 Score:                {f1[bi]:.4f}")
    print(f"  ROC-AUC:                 {roc_auc:.4f}")
    print(f"  Average Precision (AP):  {ap:.4f}")
    print("=" * 45 + "\n")

    print("  Plotting...", flush=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].plot(thresh, acc, label="Accuracy")
    ax[0, 0].plot(thresh, prec, label="Precision")
    ax[0, 0].plot(thresh, rec, label="Recall")
    ax[0, 0].plot(thresh, f1, label="F1 Score")
    ax[0, 0].axvline(bt, color="gray", ls="--", alpha=0.5)
    ax[0, 0].set(xlabel="Threshold", ylabel="Score", title="Metrics vs Threshold")
    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(fpr, tpr, lw=2, label=f"ROC (AUC={roc_auc:.4f})")
    ax[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[0, 1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(rec_curve, prc, lw=2, label=f"PR (AP={ap:.4f})")
    ax[1, 0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax[1, 0].legend()
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].hist(
        scores[labels == 1],
        bins=80, alpha=0.6, label="Positive", color="green", density=True,
    )
    ax[1, 1].hist(
        scores[labels == 0],
        bins=80, alpha=0.6, label="Negative", color="red", density=True,
    )
    ax[1, 1].axvline(bt, color="gray", ls="--", alpha=0.5, label=f"Threshold={bt:.2f}")
    ax[1, 1].set(
        xlabel="Cosine Similarity", ylabel="Density", title="Score Distribution"
    )
    ax[1, 1].legend()
    ax[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "evaluation_results.png"), dpi=150)
    plt.close()

    print(f"  Plots saved to {output_dir}/")


def main():
    args = sys.argv[1:]
    db_path = args[0] if args else "AMD64multipleCompilers.db"
    model_dir = (
        args[1]
        if len(args) > 1
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    )
    output_dir = args[2] if len(args) > 2 else "."
    max_pairs = int(args[3]) if len(args) > 3 else None
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/5] Loading model ({DEVICE})...")
    safe, normalizer = load_model(model_dir)

    print(f"[2/5] Loading pairs from {db_path}...")
    true_pairs, false_pairs = get_test_pairs(db_path, max_pairs)
    if max_pairs and len(true_pairs) > max_pairs:
        random.seed(42)
        true_pairs = random.sample(true_pairs, max_pairs)
    print(f"  {len(true_pairs):,} true + {len(false_pairs):,} false")

    fids = sorted(set(i for p in true_pairs + false_pairs for i in p))
    print(f"  {len(fids):,} unique functions")

    print(f"[3/5] Loading instructions...")
    instr = load_instructions(db_path, fids)
    print(f"  {len(instr):,} loaded")

    print(f"[4/5] Computing embeddings...")
    embeddings = compute_embeddings(safe, normalizer, instr)

    print(f"[5/5] Evaluating pairs...")
    scores, labels = evaluate_pairs(embeddings, true_pairs, false_pairs)

    print(f"\nGenerating plots...")
    plot_metrics(scores, labels, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
