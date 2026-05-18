#!/usr/bin/env python3
"""Compare evaluation results between two databases using only common functions."""

import argparse, sqlite3, json, os, sys, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetorch.safe_network import SAFE
from utils.db import load_instructions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_intersection(db_paths):
    """Find functions that exist in all databases. Return dicts mapping key->id per DB."""
    all_keys = []
    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT project, file_name, optimization, function_name, id FROM functions")
        keys = {}
        for row in cur.fetchall():
            keys[(row[0], row[1], row[2], row[3])] = row[4]
        all_keys.append(keys)
        conn.close()
    common = set(all_keys[0].keys())
    for k in all_keys[1:]:
        common &= set(k.keys())
    mappings = [{k: d[k] for k in common} for d in all_keys]
    return mappings, list(common)


def filter_pairs(db_path, valid_ids):
    """Load pairs where both ids are in valid_ids."""
    print(f"  Loading pairs from {db_path}...", flush=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id1,id2,label FROM pairs")
    true_pairs = []
    false_pairs = []
    for row in tqdm(cur, desc="Pairs"):
        if row[0] in valid_ids and row[1] in valid_ids:
            if row[2] == 1:
                true_pairs.append([row[0], row[1]])
            else:
                false_pairs.append([row[0], row[1]])
    conn.close()
    return true_pairs, false_pairs


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
        scores.append(torch.cosine_similarity(embeddings[pair[0]], embeddings[pair[1]]).item())
        labels.append(1)
    for pair in tqdm(false_pairs, desc="False pairs"):
        scores.append(torch.cosine_similarity(embeddings[pair[0]], embeddings[pair[1]]).item())
        labels.append(0)
    return np.array(scores), np.array(labels)


def compute_metrics(scores, labels):
    n = len(scores)
    thresh = np.linspace(0.01, 0.99, 200)
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
    pr = prec + rec
    f1 = np.divide(2 * prec * rec, pr, where=pr > 0, out=np.zeros(200))
    bi = np.argmax(f1)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    roc_curve_out = (fpr, tpr)
    prc, rec_curve, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    return {
        "threshold": thresh[bi],
        "accuracy": acc[bi],
        "precision": prec[bi],
        "recall": rec[bi],
        "f1": f1[bi],
        "roc_auc": roc_auc,
        "ap": ap,
        "acc_curve": (thresh, acc),
        "prec_curve": (thresh, prec),
        "rec_curve": (thresh, rec),
        "f1_curve": (thresh, f1),
        "roc_curve": roc_curve_out,
        "pr_curve": (rec_curve, prc),
        "scores": scores,
        "labels": labels,
    }


def plot_comparison(results, labels, output_dir):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (name, r) in enumerate(zip(labels, results)):
        c = colors[idx]
        axes[0, 0].plot(r["acc_curve"][0], r["acc_curve"][1], c=c, ls="-", label=f"{name} Acc")
        axes[0, 0].plot(r["prec_curve"][0], r["prec_curve"][1], c=c, ls="--", label=f"{name} Prec")
        axes[0, 0].plot(r["f1_curve"][0], r["f1_curve"][1], c=c, ls=":", label=f"{name} F1")
        axes[0, 0].axvline(r["threshold"], c=c, alpha=0.3, ls="--")

        fpr, tpr = r["roc_curve"]
        axes[0, 1].plot(fpr, tpr, c=c, lw=2, label=f"{name} (AUC={r['roc_auc']:.4f})")

        rc, pr = r["pr_curve"]
        axes[1, 0].plot(rc, pr, c=c, lw=2, label=f"{name} (AP={r['ap']:.4f})")

        axes[1, 1].hist(r["scores"][r["labels"] == 1], bins=80, alpha=0.4, density=True,
                        color=c, label=f"{name} pos")
        axes[1, 1].hist(r["scores"][r["labels"] == 0], bins=80, alpha=0.2, density=True,
                        color=c, label=f"{name} neg")

    axes[0, 0].set(xlabel="Threshold", ylabel="Score", title="Metrics vs Threshold")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set(xlabel="Cosine Similarity", ylabel="Density", title="Score Distribution")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_results.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"\nComparison plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results between databases")
    parser.add_argument("db_paths", nargs="+", help="Database paths (at least 2)")
    parser.add_argument("--model-dir", default=".", help="Model directory")
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args()

    if len(args.db_paths) < 2:
        parser.error("Need at least 2 databases to compare")

    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.model_dir, "SAFEtorch.pt")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"[1/5] Loading model ({DEVICE})...")
    safe, normalizer = SAFE.load(args.model_dir, DEVICE)

    print(f"[2/5] Finding intersection functions across {len(args.db_paths)} DBs...")
    mappings, common_keys = get_intersection(args.db_paths)
    print(f"  {len(common_keys):,} common functions")

    results = []
    for idx, (db_path, id_map) in enumerate(zip(args.db_paths, mappings)):
        name = os.path.splitext(os.path.basename(db_path))[0]
        print(f"\n{'=' * 50}")
        print(f"Processing {name}")
        print(f"{'=' * 50}")

        valid_ids = set(id_map.values())
        print(f"[3/5] Filtering pairs for {len(valid_ids):,} valid function IDs...")
        true_pairs, false_pairs = filter_pairs(db_path, valid_ids)
        if len(true_pairs) > len(false_pairs):
            random.seed(42)
            false_pairs = random.sample(false_pairs, len(true_pairs))
        elif len(false_pairs) > len(true_pairs):
            max_false = len(true_pairs)
            random.seed(42)
            false_pairs = random.sample(false_pairs, max_false)
        print(f"  {len(true_pairs):,} true + {len(false_pairs):,} false pairs")

        fids = sorted(set(i for p in true_pairs + false_pairs for i in p))
        print(f"  {len(fids):,} unique function IDs in pairs")

        print(f"  Loading instructions...")
        instr = load_instructions(db_path, fids)
        print(f"  {len(instr):,} loaded")

        print(f"  Computing embeddings...")
        embeddings = compute_embeddings(safe, normalizer, instr)

        print(f"  Evaluating pairs...")
        scores, labels = evaluate_pairs(embeddings, true_pairs, false_pairs)

        print(f"  Computing metrics...")
        result = compute_metrics(scores, labels)
        results.append(result)

        print(f"\n  Results for {name}:")
        print(f"    Best threshold:  {result['threshold']:.4f}")
        print(f"    Accuracy:        {result['accuracy']:.4f}")
        print(f"    Precision:       {result['precision']:.4f}")
        print(f"    Recall:          {result['recall']:.4f}")
        print(f"    F1 Score:        {result['f1']:.4f}")
        print(f"    ROC-AUC:         {result['roc_auc']:.4f}")
        print(f"    AP:              {result['ap']:.4f}")
        print(f"    Pairs evaluated: {len(scores):,}")

    names = [os.path.splitext(os.path.basename(p))[0] for p in args.db_paths]
    print(f"\n{'=' * 60}")
    print(f"{'Metric':<20}", end="")
    for n in names:
        print(f"{n:<20}", end="")
    print()
    print(f"{'-' * 60}")
    metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc", "ap"]
    for m in metrics_list:
        print(f"{m:<20}", end="")
        for r in results:
            print(f"{r[m]:<20.4f}", end="")
        print()
    print(f"{'=' * 60}")

    plot_comparison(results, names, args.output)


if __name__ == "__main__":
    main()
