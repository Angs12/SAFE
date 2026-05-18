#!/usr/bin/env python3
"""Compare cosine similarity: original O0-O1 vs recompiled O0-O1 for the same function."""

import sqlite3, json, sys, os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetorch.safe_network import SAFE
from safetorch.parameters import Config
from utils.function_normalizer import FunctionNormalizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_INSTRUCTIONS = 150

w2id_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "word2id.json")
with open(w2id_path) as f:
    w2id = json.load(f)
id2word = {v: k for k, v in w2id.items()}

# Load model
print(f"Loading model ({DEVICE})...")
safe = SAFE(Config())
safe.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "SAFEtorch.pt"), map_location=DEVICE))
safe = safe.to(DEVICE).eval()
normalizer = FunctionNormalizer(MAX_INSTRUCTIONS)

conn_og = sqlite3.connect("/home/tovpr/Documents/SAFE/small_test_og.db")
conn_new = sqlite3.connect("/home/tovpr/Documents/SAFE/small_test.db")
cog = conn_og.cursor()
cnew = conn_new.cursor()

def get_embedding(token_ids):
    norm, lens = normalizer.normalize_functions([token_ids])
    with torch.no_grad():
        emb = safe(torch.LongTensor(norm[0]).to(DEVICE), torch.LongTensor(lens)).detach().cpu()
    return emb

def get_tokens(cur, fid):
    cur.execute("SELECT instructions_list FROM filtered_functions WHERE id=?", (fid,))
    return json.loads(cur.fetchone()[0])

# Find functions with O0 and O1 in both DBs
cog.execute("""
    SELECT project, file_name, function_name
    FROM functions
    WHERE num_instructions BETWEEN 6 AND 80
      AND function_name NOT LIKE 'imp.%' AND function_name NOT LIKE '_GLOBAL%'
    GROUP BY project, file_name, function_name
""")

results = []
for row in cog.fetchall():
    proj, fname, fnname = row
    cnew.execute("SELECT optimization, id FROM functions WHERE project=? AND file_name=? AND function_name=?", (proj, fname, fnname))
    new = {r[0]: r[1] for r in cnew.fetchall()}
    cog.execute("SELECT optimization, id FROM functions WHERE project=? AND file_name=? AND function_name=?", (proj, fname, fnname))
    og = {r[0]: r[1] for r in cog.fetchall()}

    if "O0" in og and "O1" in og and "O0" in new and "O1" in new:
        og_o0 = get_tokens(cog, og["O0"])
        og_o1 = get_tokens(cog, og["O1"])
        new_o0 = get_tokens(cnew, new["O0"])
        new_o1 = get_tokens(cnew, new["O1"])

        # Compute embeddings
        emb_og_o0 = get_embedding(og_o0)
        emb_og_o1 = get_embedding(og_o1)
        emb_new_o0 = get_embedding(new_o0)
        emb_new_o1 = get_embedding(new_o1)

        # Cosine similarities
        sim_og = torch.cosine_similarity(emb_og_o0, emb_og_o1).item()
        sim_new = torch.cosine_similarity(emb_new_o0, emb_new_o1).item()
        sim_cross_o0 = torch.cosine_similarity(emb_og_o0, emb_new_o0).item()
        sim_cross_o1 = torch.cosine_similarity(emb_og_o1, emb_new_o1).item()

        results.append((fnname, sim_og, sim_new, sim_cross_o0, sim_cross_o1, proj, fname,
                        len(og_o0), len(og_o1), len(new_o0), len(new_o1)))

        if len(results) >= 20:
            break

# Print results
print(f"\n{'Function':<65s} {'Orig O0-O1':>10s} {'Recomp O0-O1':>12s} {'Cross O0':>9s} {'Cross O1':>9s} {'ni_og':>5s} {'ni_new':>5s}")
print("-" * 120)
for r in results:
    fnname, sim_og, sim_new, cross_o0, cross_o1, proj, fname, ni_og0, ni_og1, ni_new0, ni_new1 = r
    fn_short = fnname[:62] + ".." if len(fnname) > 64 else fnname
    print(f"{fn_short:<65s} {sim_og:>10.4f} {sim_new:>12.4f} {cross_o0:>9.4f} {cross_o1:>9.4f} {ni_og0:>3d}/{ni_og1:<3d} {ni_new0:>3d}/{ni_new1:<3d}")

if results:
    avg_og = np.mean([r[1] for r in results])
    avg_new = np.mean([r[2] for r in results])
    avg_cross_o0 = np.mean([r[3] for r in results])
    avg_cross_o1 = np.mean([r[4] for r in results])
    print("-" * 120)
    print(f"{'AVERAGE':<65s} {avg_og:>10.4f} {avg_new:>12.4f} {avg_cross_o0:>9.4f} {avg_cross_o1:>9.4f}")

conn_og.close()
conn_new.close()
