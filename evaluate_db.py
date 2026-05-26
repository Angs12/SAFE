#!/usr/bin/env python3
import sqlite3, os, sys, argparse
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

from safetorch.safe_network import SAFE
from utils.db import load_instructions
from utils.evaluation import Evaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_INSTRUCTIONS = 150


def get_test_pairs(db_path, max_false=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id1,id2 FROM pairs WHERE label=1")
    true_pairs = [list(r) for r in cur.fetchall()]
    if max_false and max_false > 0:
        cur.execute("SELECT id1,id2 FROM pairs WHERE label=0 LIMIT ?", (max_false,))
    else:
        cur.execute("SELECT id1,id2 FROM pairs WHERE label=0")
    false_pairs = [list(r) for r in cur.fetchall()]
    conn.close()
    return true_pairs, false_pairs


def plot_metrics(scores, labels, metrics, output_dir):
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
    f1 = np.divide(2 * prec * rec, prec + rec, where=(prec + rec) > 0, out=np.zeros(200))

    fpr, tpr, _ = roc_curve(labels, scores)
    prc, rec_curve, _ = precision_recall_curve(labels, scores)
    bt = metrics["best_threshold"]

    print("\n" + "=" * 45)
    print(f"  Best threshold:          {bt:.4f}")
    print(f"  Accuracy:                {metrics['accuracy']:.4f}")
    print(f"  Precision:               {metrics['precision']:.4f}")
    print(f"  Recall:                  {metrics['recall']:.4f}")
    print(f"  F1 Score:                {metrics['f1']:.4f}")
    print(f"  ROC-AUC:                 {metrics['roc_auc']:.4f}")
    print(f"  Average Precision (AP):  {metrics['average_precision']:.4f}")
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

    ax[0, 1].plot(fpr, tpr, lw=2, label=f"ROC (AUC={metrics['roc_auc']:.4f})")
    ax[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[0, 1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(rec_curve, prc, lw=2, label=f"PR (AP={metrics['average_precision']:.4f})")
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
    parser = argparse.ArgumentParser(description="Evaluate SAFE model on a pair database")
    parser.add_argument("db_path")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--output", default=".")
    parser.add_argument("--max-false", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    evaluator = Evaluator(DEVICE, MAX_INSTRUCTIONS)

    print(f"[1/5] Loading model ({DEVICE})...")
    safe, _ = SAFE.load(args.model_dir, DEVICE)

    print(f"[2/5] Loading pairs from {args.db_path}...")
    true_pairs, false_pairs = get_test_pairs(args.db_path, args.max_false)
    pairs = [(a, b, 1) for a, b in true_pairs] + [(a, b, 0) for a, b in false_pairs]
    print(f"  {len(true_pairs):,} true + {len(false_pairs):,} false")

    fids = sorted(set(i for p in pairs for i in p[:2]))
    print(f"  {len(fids):,} unique functions")

    print(f"[3/5] Loading instructions...")
    instr = load_instructions(args.db_path, fids)
    print(f"  {len(instr):,} loaded")

    print(f"[4/5] Computing embeddings...")
    embeddings = evaluator.embed_all(safe, instr, args.batch_size)

    print(f"[5/5] Evaluating pairs...")
    scores, labels = evaluator.score_pairs(embeddings, pairs)
    metrics = evaluator.compute_metrics(scores, labels)

    plot_metrics(scores, labels, metrics, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
