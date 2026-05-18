import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.function_normalizer import FunctionNormalizer


class Evaluator:
    def __init__(self, device="cpu", max_instructions=150):
        self.device = device
        self.normalizer = FunctionNormalizer(max_instructions)

    def embed_all(self, safe, instr, batch_size=64):
        safe.eval()
        items = list(instr.items())
        embs = {}
        with torch.no_grad():
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                fids = [fid for fid, _ in batch]
                seqs = [ids for _, ids in batch]
                norm, lens = self.normalizer.normalize_functions(seqs)
                out = safe(
                    torch.LongTensor(np.array(norm)).to(self.device),
                    torch.LongTensor(lens).to(self.device),
                ).detach().cpu()
                for fid, emb in zip(fids, out):
                    embs[fid] = emb
        safe.train()
        return embs

    def score_pairs(self, embeddings, pairs, batch_size=4096):
        scores, labels = [], []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            emb_a = torch.stack([embeddings[a] for a, b, _ in batch])
            emb_b = torch.stack([embeddings[b] for a, b, _ in batch])
            sim = torch.cosine_similarity(emb_a, emb_b)
            scores.extend(sim.tolist())
            labels.extend([l for _, _, l in batch])
        return np.array(scores), np.array(labels)

    def compute_metrics(self, scores, labels):
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
        best_idx = np.argmax(f1)
        roc_auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0
        ap = average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0
        return {
            "best_threshold": thresh[best_idx],
            "accuracy": acc[best_idx],
            "precision": prec[best_idx],
            "recall": rec[best_idx],
            "f1": f1[best_idx],
            "roc_auc": roc_auc,
            "average_precision": ap,
        }
