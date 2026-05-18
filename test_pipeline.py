#!/usr/bin/env python3
"""End-to-end pipeline test: synthetic DB creation → evaluation → finetuning."""

import sqlite3, json, os, sys, subprocess, tempfile, shutil
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetorch.parameters import Config

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
MAX_EMBED_ID = Config().num_embeddings - 1  # 527682
MAX_INSTRUCTIONS = 150


def create_synthetic_db(db_path):
    """Create a small synthetic database with functions and pairs."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        PRAGMA synchronous=OFF;
        CREATE TABLE functions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT, compiler TEXT, optimization TEXT,
            file_name TEXT, function_name TEXT,
            asm TEXT, num_instructions INTEGER);
        CREATE TABLE filtered_functions (
            id INTEGER PRIMARY KEY, instructions_list TEXT);
        CREATE TABLE pairs (
            id1 INTEGER NOT NULL, id2 INTEGER NOT NULL, label INTEGER NOT NULL);
        CREATE INDEX idx_func_group ON functions(project, file_name, optimization);
    """)

    rng = random.Random(42)

    def make_instructions():
        length = rng.randint(5, MAX_INSTRUCTIONS)
        return [rng.randint(1, MAX_EMBED_ID) for _ in range(length)]

    fid = 1
    projects = ["test_proj"]
    file_names = ["test_file_a", "test_file_b"]
    opts = ["O0", "O1"]
    func_names = [f"func_{i}" for i in range(4)]

    for proj in projects:
        for fname in file_names:
            for opt in opts:
                for fn in func_names:
                    instrs = make_instructions()
                    cur.execute(
                        "INSERT INTO functions VALUES (NULL,?,?,?,?,?,?,?)",
                        (proj, "gcc", opt, fname, fn, "", len(instrs)),
                    )
                    cur.execute(
                        "INSERT INTO filtered_functions VALUES (?,?)",
                        (fid, json.dumps(instrs)),
                    )
                    fid += 1

    # True pairs: same proj, fname, function_name; different optimization
    for proj in projects:
        for fname in file_names:
            cur.execute("""
                INSERT INTO pairs (id1, id2, label)
                SELECT CASE WHEN a.id < b.id THEN a.id ELSE b.id END,
                       CASE WHEN a.id < b.id THEN b.id ELSE a.id END, 1
                FROM functions a, functions b
                WHERE a.project=? AND a.file_name=?
                  AND b.project=? AND b.file_name=?
                  AND a.function_name=b.function_name
                  AND a.optimization='O0' AND b.optimization='O1'
            """, (proj, fname, proj, fname))

    # False pairs: different function_name within same group
    id_pool = []
    cur.execute("SELECT id, function_name FROM functions ORDER BY id")
    for row in cur.fetchall():
        id_pool.append(row)

    false_added = 0
    for i in range(len(id_pool)):
        for j in range(i + 1, len(id_pool)):
            id1, fn1 = id_pool[i]
            id2, fn2 = id_pool[j]
            if fn1 != fn2:
                cur.execute(
                    "INSERT INTO pairs VALUES (?,?,0)",
                    (min(id1, id2), max(id1, id2)),
                )
                false_added += 1

    conn.commit()

    n_true = cur.execute("SELECT COUNT(*) FROM pairs WHERE label=1").fetchone()[0]
    n_false = cur.execute("SELECT COUNT(*) FROM pairs WHERE label=0").fetchone()[0]
    conn.close()
    print(f"Created synthetic DB: {fid-1} functions, {n_true} true pairs, {n_false} false pairs")
    return True


def run_test():
    tmpdir = tempfile.mkdtemp(prefix="safe_test_")
    db_path = os.path.join(tmpdir, "test.db")

    try:
        # Step 1: Create synthetic database
        print("=" * 60)
        print("Step 1: Creating synthetic database...")
        print("=" * 60)
        create_synthetic_db(db_path)

        # Step 2: Run evaluate_db.py
        print("\n" + "=" * 60)
        print("Step 2: Running evaluate_db.py...")
        print("=" * 60)
        eval_result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_DIR, "evaluate_db.py"),
             db_path, "--model-dir", MODEL_DIR, "--output", tmpdir,
             "--max-false", "10", "--batch-size", "64"],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        )
        print(eval_result.stdout)
        if eval_result.stderr:
            print("STDERR:", eval_result.stderr[:500], file=sys.stderr)
        if eval_result.returncode != 0:
            print(f"evaluate_db.py FAILED with code {eval_result.returncode}", file=sys.stderr)
            return False
        if "F1 Score" not in eval_result.stdout:
            print("evaluate_db.py: metrics not found in output", file=sys.stderr)
            return False
        print("evaluate_db.py PASSED")

        # Step 3: Run finetune_safe.py (1 epoch, no val split)
        print("\n" + "=" * 60)
        print("Step 3: Running finetune_safe.py...")
        print("=" * 60)
        output_model = os.path.join(tmpdir, "finetuned.pt")
        ft_result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_DIR, "finetune_safe.py"),
             db_path, "--model-dir", MODEL_DIR, "--output", output_model,
             "--max-false", "10", "--epochs", "1", "--batch-size", "512",
             "--val-split", "0.0"],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        )
        print(ft_result.stdout)
        if ft_result.stderr:
            print("STDERR:", ft_result.stderr[:500], file=sys.stderr)
        if ft_result.returncode != 0:
            print(f"finetune_safe.py FAILED with code {ft_result.returncode}", file=sys.stderr)
            return False
        if not os.path.exists(output_model):
            print(f"finetune_safe.py: output model not found at {output_model}", file=sys.stderr)
            return False
        print("finetune_safe.py PASSED")

        # Step 4: finetune_safe.py with val split
        print("\n" + "=" * 60)
        print("Step 4: Running finetune_safe.py with validation split...")
        print("=" * 60)
        output_model2 = os.path.join(tmpdir, "finetuned_val.pt")
        ft2_result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_DIR, "finetune_safe.py"),
             db_path, "--model-dir", MODEL_DIR, "--output", output_model2,
             "--max-false", "10", "--epochs", "2", "--batch-size", "512",
             "--val-split", "0.3"],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        )
        print(ft2_result.stdout)
        if ft2_result.stderr:
            print("STDERR:", ft2_result.stderr[:500], file=sys.stderr)
        if ft2_result.returncode != 0:
            print(f"finetune_safe.py (val) FAILED with code {ft2_result.returncode}", file=sys.stderr)
            return False
        if not os.path.exists(output_model2):
            print(f"finetune_safe.py (val): output not found", file=sys.stderr)
            return False
        if "val_f1" not in ft2_result.stdout:
            print("finetune_safe.py (val): val metrics not found", file=sys.stderr)
            return False
        print("finetune_safe.py (val) PASSED")

        print("\n" + "=" * 60)
        print("ALL PIPELINE TESTS PASSED")
        print("=" * 60)
        return True

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
