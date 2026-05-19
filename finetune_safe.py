#!/usr/bin/env python3
"""Fine-tune SAFE model on a pair database using contrastive learning."""

import sqlite3, json, os, sys, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetorch.safe_network import SAFE
from utils.db import load_instructions
from utils.function_normalizer import FunctionNormalizer
from utils.evaluation import Evaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_INSTRUCTIONS = 150
MARGIN = 0.5


class PairDataset:
    def __init__(self, db_path, max_false=None, val_split=0.0):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT DISTINCT project, file_name FROM functions")
        groups = cur.fetchall()
        random.Random(42).shuffle(groups)
        n_val = max(1, int(len(groups) * val_split))
        val_groups = set(groups[:n_val])
        train_groups = groups[n_val:]
        print(f"  {len(train_groups)} train groups, {len(val_groups)} val groups")

        cur.execute("CREATE TEMP TABLE _fid_split (id INTEGER PRIMARY KEY, split INTEGER)")
        for g_start in range(0, len(train_groups), 100):
            chunk = train_groups[g_start:g_start + 100]
            cases = " OR ".join(["(project=? AND file_name=?)" for _ in chunk])
            params = [v for g in chunk for v in g]
            cur.execute(
                f"INSERT INTO _fid_split SELECT id, 0 FROM functions WHERE {cases}",
                params,
            )
        conn.commit()
        for proj, fname in val_groups:
            cur.execute(
                "INSERT INTO _fid_split SELECT id, 1 FROM functions WHERE project=? AND file_name=?",
                (proj, fname),
            )
        conn.commit()

        self.train_pairs = self._load_pairs(cur, split=0, limit=max_false)
        self.val_pairs = self._load_pairs(cur, split=1)
        print(
            f"  Train: {sum(l for _,_,l in self.train_pairs):,} true + {sum(1 for _,_,l in self.train_pairs if l==0):,} false = {len(self.train_pairs):,}"
        )
        if self.val_pairs:
            print(
                f"  Val:   {sum(l for _,_,l in self.val_pairs):,} true + {sum(1 for _,_,l in self.val_pairs if l==0):,} false = {len(self.val_pairs):,}"
            )

        train_ids = set(fid for a, b, _ in self.train_pairs for fid in (a, b))
        val_ids = set(fid for a, b, _ in self.val_pairs for fid in (a, b))
        self.train_instr = load_instructions(db_path, train_ids)
        self.val_instr = load_instructions(db_path, val_ids)
        conn.close()

    def _load_pairs(self, cur, split, limit=None):
        pair_sql = """
            SELECT p.id1, p.id2, p.label FROM pairs p
            WHERE EXISTS (SELECT 1 FROM _fid_split WHERE id=p.id1 AND split=?)
              AND EXISTS (SELECT 1 FROM _fid_split WHERE id=p.id2 AND split=?)
        """

        if limit:
            cur.execute(pair_sql + " AND label=1", (split, split))
            true_pairs = [(a, b) for a, b, _ in cur.fetchall()]
            cur.execute(
                pair_sql + " AND label=0 ORDER BY RANDOM() LIMIT ?",
                (split, split, limit),
            )
            false_pairs = [(a, b) for a, b, _ in cur.fetchall()]
        else:
            cur.execute(pair_sql, (split, split))
            all_rows = cur.fetchall()
            true_pairs = [(a, b) for a, b, l in all_rows if l == 1]
            false_pairs = [(a, b) for a, b, l in all_rows if l == 0]
            if len(false_pairs) > len(true_pairs):
                random.Random(42).shuffle(false_pairs)
                false_pairs = false_pairs[:len(true_pairs)]

        pairs = [(a, b, 1) for a, b in true_pairs] + [(a, b, 0) for a, b in false_pairs]
        random.Random(42).shuffle(pairs)
        return pairs

    def evaluate(self, safe, evaluator):
        if not self.val_pairs:
            return {}
        embs = evaluator.embed_all(safe, self.val_instr)
        scores, labels = evaluator.score_pairs(embs, self.val_pairs)
        return evaluator.compute_metrics(scores, labels)


def _unwrap_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def embed_batch(normalizer, safe, seqs):
    norm, lens = normalizer.normalize_functions(seqs)
    with torch.set_grad_enabled(True):
        return safe(
            torch.LongTensor(np.array(norm)).to(DEVICE),
            torch.LongTensor(lens).to(DEVICE),
        )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAFE model on paired function data")
    parser.add_argument("db_path")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-false", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.05)
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.model_dir, "SAFEtorch_finetuned.pt")

    print(f"[1/4] Loading model ({DEVICE})...")
    safe, normalizer = SAFE.load(args.model_dir, DEVICE, train=True)
    frozen = sum(p.numel() for p in safe.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in safe.parameters() if p.requires_grad)
    print(f"  {frozen:,} frozen (embedding), {trainable:,} trainable (RNN+attention+dense)")

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        gpu_ids = list(range(min(2, n_gpu)))
        safe = nn.DataParallel(safe, device_ids=gpu_ids)
        print(f"  Using {len(gpu_ids)} GPUs: {gpu_ids}")
    else:
        print(f"  Using {DEVICE}" + (f" ({n_gpu} GPU)" if n_gpu == 1 else ""))

    print(f"[2/4] Loading pairs from {args.db_path}...")
    dataset = PairDataset(args.db_path, args.max_false, args.val_split)
    n = len(dataset.train_pairs)
    evaluator = Evaluator(DEVICE, MAX_INSTRUCTIONS)

    print(f"[3/4] Fine-tuning ({args.epochs} epochs, batch_size={args.batch_size})...")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, safe.parameters()), lr=1e-5
    )
    best_f1 = 0.0

    for epoch in range(args.epochs):
        random.shuffle(dataset.train_pairs)
        total_loss = n_batches = 0
        t0 = time.time()

        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            batch_pairs = dataset.train_pairs[start:end]
            fids = list(set(fid for a, b, _ in batch_pairs for fid in (a, b)))

            seqs = [dataset.train_instr[fid] for fid in fids]
            embs = embed_batch(normalizer, safe, seqs)
            fid_to_emb = dict(zip(fids, embs))

            emb_a = torch.stack([fid_to_emb[a] for a, b, _ in batch_pairs])
            emb_b = torch.stack([fid_to_emb[b] for a, b, _ in batch_pairs])
            labels = torch.FloatTensor([l for _, _, l in batch_pairs]).to(DEVICE)

            sim = torch.cosine_similarity(emb_a, emb_b)
            d = 1 - sim
            loss = (
                labels * d.pow(2) + (1 - labels) * torch.clamp(MARGIN - d, min=0).pow(2)
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 25 == 0:
                elapsed = time.time() - t0
                remaining = (elapsed / n_batches) * ((n // args.batch_size + 1) - n_batches)
                print(
                    f"  Ep {epoch+1}/{args.epochs}  batch {n_batches}/{n//args.batch_size+1}  loss={total_loss/n_batches:.6f}  {elapsed:.0f}s  est {remaining:.0f}s",
                    flush=True,
                )

        line = f"  Epoch {epoch+1}/{args.epochs}  avg_loss={total_loss/n_batches:.6f}  [{time.time()-t0:.0f}s]"

        if args.val_split:
            metrics = dataset.evaluate(safe, evaluator)
            if metrics:
                line += f"  val_f1={metrics['f1']:.4f}  val_acc={metrics['accuracy']:.4f}  val_auc={metrics['roc_auc']:.4f}"
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    torch.save(_unwrap_state_dict(safe), output_path)
                    line += "  *saved*"
        else:
            torch.save(_unwrap_state_dict(safe), output_path)

        print(line)

    if args.val_split and best_f1 == 0.0:
        torch.save(_unwrap_state_dict(safe), output_path)
    print(f"[4/4] Model saved to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
