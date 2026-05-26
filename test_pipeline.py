#!/usr/bin/env python3
"""End-to-end pipeline test: synthetic DB creation → evaluation → finetuning."""

import sqlite3, json, os, sys, subprocess, tempfile, shutil
import random

from safetorch.parameters import Config
from create_dataset import split_pairs_train_val

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
MAX_EMBED_ID = Config().num_embeddings - 1
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

    id_pool = []
    cur.execute("SELECT id, function_name FROM functions ORDER BY id")
    for row in cur.fetchall():
        id_pool.append(row)

    for i in range(len(id_pool)):
        for j in range(i + 1, len(id_pool)):
            id1, fn1 = id_pool[i]
            id2, fn2 = id_pool[j]
            if fn1 != fn2:
                cur.execute(
                    "INSERT INTO pairs VALUES (?,?,0)",
                    (min(id1, id2), max(id1, id2)),
                )

    conn.commit()

    n_true = cur.execute("SELECT COUNT(*) FROM pairs WHERE label=1").fetchone()[0]
    n_false = cur.execute("SELECT COUNT(*) FROM pairs WHERE label=0").fetchone()[0]
    conn.close()
    print(f"Created synthetic DB: {fid-1} functions, {n_true} true pairs, {n_false} false pairs")


def run_test():
    tmpdir = tempfile.mkdtemp(prefix="safe_test_")
    db_test = os.path.join(tmpdir, "test.db")
    db_train = os.path.join(tmpdir, "train.db")

    try:
        # Step 1: Test-mode DB → evaluate_db.py
        print("=" * 60)
        print("Step 1: Creating test DB...")
        print("=" * 60)
        create_synthetic_db(db_test)

        eval_result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_DIR, "evaluate_db.py"),
             db_test, "--model-dir", MODEL_DIR, "--output", tmpdir,
             "--max-false", "10", "--batch-size", "64"],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        )
        print(eval_result.stdout)
        if eval_result.stderr:
            print("STDERR:", eval_result.stderr[:500], file=sys.stderr)
        if eval_result.returncode != 0:
            print(f"evaluate_db.py FAILED", file=sys.stderr)
            return False
        if "F1 Score" not in eval_result.stdout:
            print("evaluate_db.py: metrics not found", file=sys.stderr)
            return False
        print("evaluate_db.py PASSED")

        # Step 2: Train-mode DB → finetune_safe.py
        print("\n" + "=" * 60)
        print("Step 2: Creating train DB...")
        print("=" * 60)
        create_synthetic_db(db_train)
        conn = sqlite3.connect(db_train)
        split_pairs_train_val(conn)
        conn.close()

        output_model = os.path.join(tmpdir, "finetuned.pt")
        ft_result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_DIR, "finetune_safe.py"),
             db_train, "--model-dir", MODEL_DIR, "--output", output_model,
             "--max-false", "10", "--epochs", "1", "--batch-size", "512"],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        )
        print(ft_result.stdout)
        if ft_result.stderr:
            print("STDERR:", ft_result.stderr[:500], file=sys.stderr)
        if ft_result.returncode != 0:
            print(f"finetune_safe.py FAILED", file=sys.stderr)
            return False
        if not os.path.exists(output_model):
            print(f"finetune_safe.py: output model not found", file=sys.stderr)
            return False
        print("finetune_safe.py PASSED")

        # Step 3: finetune_safe.py with more epochs (tests best-f1 checkpointing)
        print("\n" + "=" * 60)
        print("Step 3: Running finetune_safe.py (2 epochs)...")
        print("=" * 60)
        output_model2 = os.path.join(tmpdir, "finetuned_2ep.pt")
        ft2_result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_DIR, "finetune_safe.py"),
             db_train, "--model-dir", MODEL_DIR, "--output", output_model2,
             "--max-false", "10", "--epochs", "2", "--batch-size", "512"],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        )
        print(ft2_result.stdout)
        if ft2_result.stderr:
            print("STDERR:", ft2_result.stderr[:500], file=sys.stderr)
        if ft2_result.returncode != 0:
            print(f"finetune_safe.py (2 epoch) FAILED", file=sys.stderr)
            return False
        if not os.path.exists(output_model2):
            print(f"finetune_safe.py (2 epoch): output not found", file=sys.stderr)
            return False
        if "val_f1" not in ft2_result.stdout:
            print("finetune_safe.py (2 epoch): val metrics not found", file=sys.stderr)
            return False
        print("finetune_safe.py (2 epoch) PASSED")

        print("\n" + "=" * 60)
        print("ALL PIPELINE TESTS PASSED")
        print("=" * 60)
        return True

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
