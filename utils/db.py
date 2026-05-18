import sqlite3, json


def load_instructions(db_path, function_ids):
    if not function_ids:
        return {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    instr = {}
    ids_list = list(function_ids)
    for i in range(0, len(ids_list), 900):
        batch = ids_list[i:i + 900]
        ph = ",".join("?" * len(batch))
        cur.execute(
            f"SELECT id, instructions_list FROM filtered_functions WHERE id IN ({ph})",
            batch,
        )
        for row in cur.fetchall():
            instr[row[0]] = json.loads(row[1])
    conn.close()
    return instr
