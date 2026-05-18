# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.function_normalizer import FunctionNormalizer
from utils.instructions_converter import InstructionsConverter
from utils.capstone_disassembler import disassemble
from utils.radare_analyzer import BinaryAnalyzer
from safetorch.safe_network import SAFE
from safetorch.parameters import Config
import torch

import sys
import sqlite3
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.metrics as skm

EMBEDDING_DIM = 100


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            binary_path TEXT NOT NULL,
            function_name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_binary_function 
        ON embeddings(binary_path, function_name)
    """)
    conn.commit()
    return conn


def store_embedding(conn, binary_path, fn_name, embedding):
    embedding_bytes = json.dumps(embedding.numpy().tolist())
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO embeddings (binary_path, function_name, embedding) VALUES (?, ?, ?)",
        (binary_path, fn_name, embedding_bytes),
    )
    conn.commit()


def get_function_names(conn, binary_path):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT function_name FROM embeddings WHERE binary_path = ?", (binary_path,)
    )
    return [row[0] for row in cursor.fetchall()]


def get_embedding(conn, binary_path, fn_name):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT embedding FROM embeddings WHERE binary_path = ? AND function_name = ?",
        (binary_path, fn_name),
    )
    row = cursor.fetchone()
    if row:
        return np.asarray(json.loads(row[0]))
    return None


def embed_binary_store(binary_path, safe, converter, normalizer, db_path):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM embeddings WHERE binary_path = ?",
        (binary_path,),
    )
    if cursor.fetchone():
        return
    bin = BinaryAnalyzer(binary_path)
    fns = bin.get_functions()
    for fn, addr in tqdm(fns):
        asm = bin.get_hexasm(addr)
        instructions = disassemble(asm, bin.arch, bin.bits)
        print(instructions)
        converted_instructions = converter.convert_to_ids(instructions)
        instructions, length = normalizer.normalize_functions([converted_instructions])
        tensor = torch.LongTensor(instructions[0])
        function_embedding = safe(tensor, torch.LongTensor(length)).detach()
        store_embedding(conn, binary_path, fn, function_embedding)
    conn.close()


def load_binary_embeddings(db_path, binary_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT embedding, function_name FROM embeddings WHERE binary_path = ?",
        (binary_path,),
    )
    embeddings = cursor.fetchall()
    conn.close()
    return [
        (embedding[1], np.asarray(json.loads(embedding[0]))) for embedding in embeddings
    ]


def evaluate(binary_path1, binary_path2, db_path):
    conn = sqlite3.connect(db_path)
    fns1 = load_binary_embeddings(db_path, binary_path1)
    fns2 = load_binary_embeddings(db_path, binary_path2)

    scores = [0 for _ in range(len(fns1))]
    true_labels = [0 for _ in range(len(fns1))]
    for fn1, embedding1 in tqdm(fns1):
        score = 0
        true_label = 0
        for fn2, embedding2 in fns2:
            res = torch.cosine_similarity(
                torch.from_numpy(embedding1), torch.from_numpy(embedding2)
            )
            if res > score:
                score = res
                true_label = 1 if fn1 == fn2 else 0
        scores.append(score)
        true_labels.append(true_label)

    conn.close()
    return true_labels, scores


binary_path1_og = sys.argv[1]
binary_path2_og = sys.argv[2]
if len(sys.argv) > 3:
    binary_path1_tr = sys.argv[3]
    binary_path2_tr = sys.argv[4]

config = Config()
safe = SAFE(config)

I2V_FILENAME = "model/word2id.json"
converter = InstructionsConverter(I2V_FILENAME)
normalizer = FunctionNormalizer(max_instruction=30000)

SAFE_torch_model_path = "model/SAFEtorch.pt"
state_dict = torch.load(SAFE_torch_model_path)
safe.load_state_dict(state_dict)
safe = safe.eval()

db_path = "embeddings.db"
init_db(db_path)
embed_binary_store(binary_path1_og, safe, converter, normalizer, db_path)
embed_binary_store(binary_path2_og, safe, converter, normalizer, db_path)

true_labels, scores = evaluate(binary_path1_og, binary_path2_og, db_path)

f1_scores_og = []
acc_og = []

for threshold in np.linspace(0.01, 0.99, 100):
    pred_labels = [1 if s > threshold else 0 for s in scores]
    f1_scores_og.append(skm.f1_score(true_labels, pred_labels))
    acc_og.append(skm.accuracy_score(true_labels, pred_labels))


if len(sys.argv) > 3:

    embed_binary_store(binary_path1_tr, safe, converter, normalizer, db_path)
    embed_binary_store(binary_path2_tr, safe, converter, normalizer, db_path)

    true_labels, scores = evaluate(binary_path1_tr, binary_path2_tr, db_path)

    f1_scores_tr = []
    acc_tr = []

    for threshold in np.linspace(0.01, 0.99, 100):
        pred_labels = [1 if s > threshold else 0 for s in scores]
        f1_scores_tr.append(skm.f1_score(true_labels, pred_labels))
        acc_tr.append(skm.accuracy_score(true_labels, pred_labels))

    plt.plot(f1_scores_og, label="Original f1 score")
    plt.plot(f1_scores_tr, label="Transformed f1 score")
    plt.legend()
    plt.savefig("f1_scores.png")

    plt.plot(acc_og, label="Original acc")
    plt.plot(acc_tr, label="Transformed acc")
    plt.legend()
    plt.savefig("acc.png")

    print("Max acc score ")
    print(max(acc_tr))

    print("Max F1 score")
    print(max(f1_scores_tr))

print("Max F1 score original")
print(max(f1_scores_og))
