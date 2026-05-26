#!/usr/bin/env python3
"""Compare evaluation results between two databases using only common functions."""

import argparse, sqlite3, os, sys, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm

from utils.evaluation import Evaluator
from safetorch.safe_network import SAFE
from utils.db import load_instructions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_intersection(db_paths):
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


def plot_comparison(results, labels, output_dir):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (name, r) in enumerate(zip(labels, results)):
        c = colors[idx]
        s, l = r["scores"], r["labels"]

        n = len(s)
        thresh = np.linspace(0.01, 0.99, 200)
        tp = np.zeros(200); fp = np.zeros(200); fn = np.zeros(200); tn = np.zeros(200)
        chunk = 200000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            cs = s[start:end]
            cl = l[start:end]
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
        f1 = np.divide(2 * prec * rec, prec + rec, where=(prec + rec) > 0, out=np.zeros(200))

        fpr, tpr, _ = roc_curve(l, s)
        prc, rec_curve, _ = precision_recall_curve(l, s)

        axes[0, 0].plot(thresh, acc, c=c, ls="-", label=f"{name} Acc")
        axes[0, 0].plot(thresh, prec, c=c, ls="--", label=f"{name} Prec")
        axes[0, 0].plot(thresh, f1, c=c, ls=":", label=f"{name} F1")
        axes[0, 0].axvline(r["best_threshold"], c=c, alpha=0.3, ls="--")

        axes[0, 1].plot(fpr, tpr, c=c, lw=2, label=f"{name} (AUC={r['roc_auc']:.4f})")

        axes[1, 0].plot(rec_curve, prc, c=c, lw=2, label=f"{name} (AP={r['average_precision']:.4f})")

        axes[1, 1].hist(s[l == 1], bins=80, alpha=0.4, density=True,
                        color=c, label=f"{name} pos")
        axes[1, 1].hist(s[l == 0], bins=80, alpha=0.2, density=True,
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
    safe, _ = SAFE.load(args.model_dir, DEVICE)

    print(f"[2/5] Finding intersection functions across {len(args.db_paths)} DBs...")
    mappings, common_keys = get_intersection(args.db_paths)
    print(f"  {len(common_keys):,} common functions")

    evaluator = Evaluator(DEVICE)
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

        pairs = [(a, b, 1) for a, b in true_pairs] + [(a, b, 0) for a, b in false_pairs]

        fids = sorted(set(i for p in true_pairs + false_pairs for i in p))
        print(f"  {len(fids):,} unique function IDs in pairs")

        print(f"  Loading instructions...")
        instr = load_instructions(db_path, fids)
        print(f"  {len(instr):,} loaded")

        print(f"  Computing embeddings...")
        embeddings = evaluator.embed_all(safe, instr)

        print(f"  Evaluating pairs...")
        scores, labels = evaluator.score_pairs(embeddings, pairs)

        print(f"  Computing metrics...")
        metrics = evaluator.compute_metrics(scores, labels)
        result = {"scores": scores, "labels": labels, **metrics}
        results.append(result)

        print(f"\n  Results for {name}:")
        print(f"    Best threshold:  {result['best_threshold']:.4f}")
        print(f"    Accuracy:        {result['accuracy']:.4f}")
        print(f"    Precision:       {result['precision']:.4f}")
        print(f"    Recall:          {result['recall']:.4f}")
        print(f"    F1 Score:        {result['f1']:.4f}")
        print(f"    ROC-AUC:         {result['roc_auc']:.4f}")
        print(f"    AP:              {result['average_precision']:.4f}")
        print(f"    Pairs evaluated: {len(scores):,}")

    names = [os.path.splitext(os.path.basename(p))[0] for p in args.db_paths]
    print(f"\n{'=' * 60}")
    print(f"{'Metric':<20}", end="")
    for n in names:
        print(f"{n:<20}", end="")
    print()
    print(f"{'-' * 60}")
    metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    for m in metrics_list:
        print(f"{m:<20}", end="")
        for r in results:
            print(f"{r[m]:<20.4f}", end="")
        print()
    print(f"{'=' * 60}")

    plot_comparison(results, names, args.output)


if __name__ == "__main__":
    main()
