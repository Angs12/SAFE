#!/usr/bin/env python3
"""Fine-tune SAFE model on a pair database using contrastive learning.

For CPU efficiency, only the attention + dense layers are trained (embedding and RNN are frozen).
"""

import sqlite3, json, os, sys, random, time
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetorch.safe_network import SAFE
from safetorch.parameters import Config
from utils.function_normalizer import FunctionNormalizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_INSTRUCTIONS = 150
MARGIN = 0.5


def load_model(model_dir):
    safe = SAFE(Config())
    safe.load_state_dict(torch.load(os.path.join(model_dir, "SAFEtorch.pt"), map_location=DEVICE))
    # Freeze embedding and RNN layers (only train attention + dense)
    for name, param in safe.named_parameters():
        if name.startswith("instructions_embeddings") or name.startswith("bidirectional_rnn"):
            param.requires_grad = False
    return safe.to(DEVICE).train(), FunctionNormalizer(MAX_INSTRUCTIONS)


class PairDataset:
    def __init__(self, db_path, max_pairs=20000):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM pairs WHERE label=1")
        n_true = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM pairs WHERE label=0")
        n_false = cur.fetchone()[0] or 0
        if max_pairs and n_true > max_pairs:
            n_true = max_pairs
        n_false = min(n_false, n_true)
        true_pairs = cur.execute(
            "SELECT id1, id2 FROM pairs WHERE label=1 LIMIT ?", (n_true,)
        ).fetchall()
        false_pairs = cur.execute(
            "SELECT id1, id2 FROM pairs WHERE label=0 LIMIT ?", (n_false,)
        ).fetchall()
        self.pairs = [(a, b, 1) for a, b in true_pairs] + [(a, b, 0) for a, b in false_pairs]
        random.Random(42).shuffle(self.pairs)
        print(f"  {len(true_pairs):,} true + {len(false_pairs):,} false = {len(self.pairs):,} total")
        all_ids = set(fid for a, b, _ in self.pairs for fid in (a, b))
        print(f"  {len(all_ids):,} unique function IDs")
        self.instr = {}
        for i in range(0, len(all_ids), 900):
            batch = list(all_ids)[i:i + 900]
            placeholders = ','.join('?' * len(batch))
            cur.execute(f"SELECT id, instructions_list FROM filtered_functions WHERE id IN ({placeholders})", batch)
            for row in cur.fetchall():
                self.instr[row[0]] = json.loads(row[1])
        conn.close()


def embed_batch(normalizer, safe, seqs):
    norm, lens = normalizer.normalize_functions(seqs)
    with torch.set_grad_enabled(True):
        return safe(torch.LongTensor(np.array(norm)).to(DEVICE), torch.LongTensor(lens).to(DEVICE))


def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: finetune_safe.py <db_path> [model_dir] [output_path] [max_pairs] [epochs] [batch_size]")
        sys.exit(1)

    db_path = args[0]
    model_dir = args[1] if len(args) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    output_path = args[2] if len(args) > 2 else os.path.join(model_dir, "SAFEtorch_finetuned.pt")
    max_pairs = int(args[3]) if len(args) > 3 else 10000
    epochs = int(args[4]) if len(args) > 4 else 2
    batch_size = int(args[5]) if len(args) > 5 else 128

    print(f"[1/4] Loading model ({DEVICE})...")
    safe, normalizer = load_model(model_dir)
    frozen = sum(p.numel() for p in safe.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in safe.parameters() if p.requires_grad)
    print(f"  {frozen:,} frozen (embedding+RNN), {trainable:,} trainable (attention+dense)")

    print(f"[2/4] Loading pairs from {db_path}...")
    dataset = PairDataset(db_path, max_pairs)
    n = len(dataset.pairs)

    print(f"[3/4] Fine-tuning ({epochs} epochs, batch_size={batch_size})...")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, safe.parameters()), lr=1e-5)
    indices = list(range(n))

    for epoch in range(epochs):
        random.shuffle(dataset.pairs)
        total_loss = n_batches = 0
        t0 = time.time()

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_pairs = [dataset.pairs[i] for i in indices[start:end]]
            fids = list(set(fid for a, b, _ in batch_pairs for fid in (a, b)))

            # Batch embed all unique function IDs
            seqs = [dataset.instr[fid] for fid in fids]
            embs = embed_batch(normalizer, safe, seqs)
            fid_to_emb = dict(zip(fids, embs))

            emb_a = torch.stack([fid_to_emb[a] for a, b, _ in batch_pairs])
            emb_b = torch.stack([fid_to_emb[b] for a, b, _ in batch_pairs])
            labels = torch.FloatTensor([l for _, _, l in batch_pairs]).to(DEVICE)

            sim = torch.cosine_similarity(emb_a, emb_b)
            d = 1 - sim
            loss = (labels * d.pow(2) + (1 - labels) * torch.clamp(MARGIN - d, min=0).pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 25 == 0:
                elapsed = time.time() - t0
                remaining = (elapsed / n_batches) * ((n // batch_size + 1) - n_batches)
                print(f"  Ep {epoch+1}/{epochs}  batch {n_batches}/{n//batch_size+1}  loss={total_loss/n_batches:.6f}  {elapsed:.0f}s  est {remaining:.0f}s", flush=True)

        print(f"  Epoch {epoch+1} done  avg_loss={total_loss/n_batches:.6f}  [{time.time()-t0:.0f}s]")

    print(f"[4/4] Saving fine-tuned model to {output_path}...")
    torch.save(safe.state_dict(), output_path)
    print("Done!")


if __name__ == "__main__":
    main()
