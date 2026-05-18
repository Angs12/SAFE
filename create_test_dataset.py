#!/usr/bin/env python3
import os, sys, json, re, sqlite3, random, argparse
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.capstone_disassembler import disassemble
from utils.radare_analyzer import BinaryAnalyzer

FILENAME_RE = re.compile(r"^(.+?)-(.+?)-(O0|O1|O2|O3|Os)-([a-f0-9]{32})$")
_I2ID = None  # global in worker processes


def init_worker(w2id_path):
    global _I2ID
    with open(w2id_path) as f:
        _I2ID = json.load(f)


def parse_binary_filename(name):
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return {"project": m.group(1), "file_name": m.group(2),
            "optimization": m.group(3), "hash": m.group(4), "name": name}


def convert_ids(instructions, i2id):
    ret = []
    for x in instructions:
        if x in i2id:
            ret.append(i2id[x] + 1)
        elif "X_" in x:
            ret.append(i2id["X_UNK"] + 1)
        elif "A_" in x:
            ret.append(i2id["A_UNK"] + 1)
        else:
            ret.append(i2id["X_UNK"] + 1)
    return ret


def process_single_binary(args):
    path, meta = args
    global _I2ID
    analyzer = None
    try:
        analyzer = BinaryAnalyzer(path)
    except Exception as e:
        if analyzer:
            analyzer.close()
        return {"meta": meta, "functions": [], "error": str(e)}
    if analyzer.arch is None:
        analyzer.close()
        return {"meta": meta, "functions": [], "error": "unsupported arch"}
    functions = []
    try:
        for fn_name, addr in analyzer.get_functions():
            if fn_name is None or fn_name.startswith("sub.") or fn_name.startswith("sub_") or fn_name.startswith("fcn."):
                continue
            try:
                asm_hex = analyzer.get_hexasm(addr)
                if not asm_hex:
                    continue
                instructions = disassemble(asm_hex, analyzer.arch, analyzer.bits)
                if not instructions:
                    continue
                converted = convert_ids(instructions, _I2ID)
                functions.append({"name": fn_name, "asm_hex": asm_hex,
                                  "raw_ids": converted,
                                  "num_instructions": len(converted)})
            except Exception:
                continue
    finally:
        analyzer.close()
    return {"meta": meta, "functions": functions, "error": None}


def get_binary_list(folder, max_groups=None):
    binaries = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        meta = parse_binary_filename(fname)
        if meta is not None:
            meta["path"] = path
            binaries.append(meta)
    groups = defaultdict(list)
    for b in binaries:
        groups[(b["project"], b["file_name"])].append(b)
    group_list = list(groups.values())
    random.Random(42).shuffle(group_list)
    if max_groups:
        group_list = group_list[:max_groups]
    return [(m["path"], m) for g in group_list for m in g], len(group_list)


def generate_pairs(conn):
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT project, file_name FROM functions")
    groups = cur.fetchall()

    for p, f in tqdm(groups, desc="Building pairs"):
        opts = sorted(cur.execute(
            "SELECT DISTINCT optimization FROM functions WHERE project=? AND file_name=?",
            (p, f)).fetchall())
        opts = [o[0] for o in opts]

        for i in range(len(opts)):
            for j in range(i + 1, len(opts)):
                cur.execute("""
                    INSERT INTO pair_buf (id1,id2,label)
                    SELECT CASE WHEN a.id<b.id THEN a.id ELSE b.id END,
                           CASE WHEN a.id<b.id THEN b.id ELSE a.id END,1
                    FROM functions a,functions b
                    WHERE a.project=? AND a.file_name=? AND b.project=?
                      AND b.file_name=? AND a.optimization=?
                      AND b.optimization=? AND a.function_name=b.function_name
                """, (p, f, p, f, opts[i], opts[j]))

        num_opt_pairs = len(opts) * (len(opts) - 1) // 2
        if num_opt_pairs == 0:
            continue
        total_true = cur.execute(
            "SELECT COUNT(*) FROM pair_buf WHERE label=1").fetchone()[0]
        limit = max(total_true * 5 // num_opt_pairs, 5000)

        for i in range(len(opts)):
            for j in range(i + 1, len(opts)):
                m1 = dict(cur.execute(
                    "SELECT function_name,id FROM functions "
                    "WHERE project=? AND file_name=? AND optimization=?",
                    (p, f, opts[i])).fetchall())
                m2 = dict(cur.execute(
                    "SELECT function_name,id FROM functions "
                    "WHERE project=? AND file_name=? AND optimization=?",
                    (p, f, opts[j])).fetchall())

                names1, ids1 = list(m1.keys()), list(m1.values())
                names2, ids2 = list(m2.keys()), list(m2.values())
                n1, n2 = len(ids1), len(ids2)
                same_count = sum(1 for n in names1 if n in m2)
                total_valid = n1 * n2 - same_count
                sample_size = min(limit, total_valid)

                batch = []
                if sample_size == total_valid:
                    for ni, a in zip(names1, ids1):
                        for nj, b in zip(names2, ids2):
                            if ni != nj:
                                batch.append((min(a, b), max(a, b), 0))
                elif sample_size > 0:
                    seen = set()
                    rng = random.Random(42)
                    while len(batch) < sample_size:
                        idx = rng.randrange(n1 * n2)
                        if idx in seen:
                            continue
                        seen.add(idx)
                        ii, jj = divmod(idx, n2)
                        if names1[ii] != names2[jj]:
                            a, b = ids1[ii], ids2[jj]
                            batch.append((min(a, b), max(a, b), 0))

                for k in range(0, len(batch), 5000):
                    cur.executemany(
                        "INSERT INTO pair_buf VALUES (?,?,?)",
                        batch[k:k + 5000])
        conn.commit()

    cur.execute("ALTER TABLE pair_buf RENAME TO pairs")
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary_folder")
    ap.add_argument("output_db", nargs="?", default="test_dataset.db")
    ap.add_argument("model_dir", nargs="?",
                    default=os.path.join(os.path.dirname(
                        os.path.abspath(__file__)), "model"))
    ap.add_argument("--max-groups", type=int)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count(), 2))
    args = ap.parse_args()

    w2id_path = os.path.join(args.model_dir, "word2id.json")
    if not os.path.exists(w2id_path):
        print(f"Error: {w2id_path} not found")
        sys.exit(1)

    print(f"Scanning {args.binary_folder}...")
    binary_list, n_groups = get_binary_list(args.binary_folder, args.max_groups)
    print(f"  {len(binary_list)} binaries, {n_groups} groups",
          end=" (limited)" if args.max_groups else "", sep="")
    print()

    conn = sqlite3.connect(args.output_db)
    cur = conn.cursor()
    cur.executescript("""
        PRAGMA synchronous=OFF;
        PRAGMA journal_mode=WAL;
        CREATE TABLE functions (id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT,compiler TEXT,optimization TEXT,
            file_name TEXT,function_name TEXT,
            asm TEXT,num_instructions INTEGER);
        CREATE TABLE filtered_functions (id INTEGER PRIMARY KEY,
            instructions_list TEXT);
        CREATE TABLE pair_buf (id1 INTEGER NOT NULL,id2 INTEGER NOT NULL,
            label INTEGER NOT NULL);
        CREATE INDEX idx_func_group ON functions(project,file_name,optimization);
    """)
    conn.commit()

    print(f"Processing with {args.workers} workers...")
    total_fns = 0
    errors = 0
    fid = 1

    with Pool(args.workers, initializer=init_worker,
              initargs=(w2id_path,), maxtasksperchild=50) as pool:
        for r in tqdm(pool.imap_unordered(
                process_single_binary,
                [(p, m) for p, m in binary_list],
                chunksize=max(1, len(binary_list) // (args.workers * 4))),
                total=len(binary_list), desc="Binaries"):
            if r["error"]:
                errors += 1
                continue
            m = r["meta"]
            for fn in r["functions"]:
                cur.execute(
                    "INSERT INTO functions VALUES (NULL,?,?,?,?,?,?,?)",
                    (m["project"], "unknown", m["optimization"],
                     m["file_name"], fn["name"], fn["asm_hex"],
                     fn["num_instructions"]))
                cur.execute(
                    "INSERT INTO filtered_functions VALUES (?,?)",
                    (fid, json.dumps(fn["raw_ids"])))
                fid += 1
                total_fns += 1
            if fid % 5000 == 0:
                conn.commit()

    conn.commit()
    print(f"  {total_fns:,} functions, {errors} skipped")

    print(f"Generating pairs...")
    generate_pairs(conn)

    n1 = cur.execute("SELECT COUNT(*) FROM pairs WHERE label=1").fetchone()[0]
    n0 = cur.execute("SELECT COUNT(*) FROM pairs WHERE label=0").fetchone()[0]
    conn.close()
    print(f"  True pairs: {n1:,}  False pairs: {n0:,}")
    print(f"\nDone. Run: python evaluate_db.py {args.output_db} {args.model_dir} .")


if __name__ == "__main__":
    main()
