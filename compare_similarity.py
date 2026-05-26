#!/usr/bin/env python3
"""Compare cosine similarity: original O0-O1 vs recompiled O0-O1 for the same function."""

import argparse, sqlite3, json, sys, os
import torch
import numpy as np

from safetorch.safe_network import SAFE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokens(cur, fid):
    cur.execute("SELECT instructions_list FROM filtered_functions WHERE id=?", (fid,))
    return json.loads(cur.fetchone()[0])


def main():
    parser = argparse.ArgumentParser(
        description="Compare cosine similarity between original and recompiled binaries")
    parser.add_argument("--db1", required=True, help="First (original) database")
    parser.add_argument("--db2", required=True, help="Second (recompiled) database")
    parser.add_argument("--model-dir", default=".", help="Model directory")
    parser.add_argument("--max-results", type=int, default=20, help="Maximum results to print")
    args = parser.parse_args()

    print(f"Loading model ({DEVICE})...")
    safe, normalizer = SAFE.load(args.model_dir, DEVICE)

    conn_og = sqlite3.connect(args.db1)
    conn_new = sqlite3.connect(args.db2)
    cog = conn_og.cursor()
    cnew = conn_new.cursor()

    def get_embedding(token_ids):
        norm, lens = normalizer.normalize_functions([token_ids])
        with torch.no_grad():
            emb = safe(torch.LongTensor(norm[0]).to(DEVICE), torch.LongTensor(lens)).detach().cpu()
        return emb

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
        cnew.execute("SELECT optimization, id FROM functions WHERE project=? AND file_name=? AND function_name=?",
                     (proj, fname, fnname))
        new = {r[0]: r[1] for r in cnew.fetchall()}
        cog.execute("SELECT optimization, id FROM functions WHERE project=? AND file_name=? AND function_name=?",
                    (proj, fname, fnname))
        og = {r[0]: r[1] for r in cog.fetchall()}

        if "O0" in og and "O1" in og and "O0" in new and "O1" in new:
            og_o0 = get_tokens(cog, og["O0"])
            og_o1 = get_tokens(cog, og["O1"])
            new_o0 = get_tokens(cnew, new["O0"])
            new_o1 = get_tokens(cnew, new["O1"])

            emb_og_o0 = get_embedding(og_o0)
            emb_og_o1 = get_embedding(og_o1)
            emb_new_o0 = get_embedding(new_o0)
            emb_new_o1 = get_embedding(new_o1)

            sim_og = torch.cosine_similarity(emb_og_o0, emb_og_o1).item()
            sim_new = torch.cosine_similarity(emb_new_o0, emb_new_o1).item()
            sim_cross_o0 = torch.cosine_similarity(emb_og_o0, emb_new_o0).item()
            sim_cross_o1 = torch.cosine_similarity(emb_og_o1, emb_new_o1).item()

            results.append((fnname, sim_og, sim_new, sim_cross_o0, sim_cross_o1, proj, fname,
                            len(og_o0), len(og_o1), len(new_o0), len(new_o1)))

            if len(results) >= args.max_results:
                break

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


if __name__ == "__main__":
    main()
