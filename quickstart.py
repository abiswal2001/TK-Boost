"""
TK-Boost Quick Start
====================
End-to-end example using provided local007 artifacts:
1) Generate tribal knowledge from `tkstore/example/example.json`
2) Refine existing draft SQL from `tkstore/example/execution_query.sql`
3) Compare agent vs refined vs gold outputs
"""

from pathlib import Path

import tkboost
from tkboost import SQLiteExecutor

EXAMPLE_JSON = "tkstore/example/example.json"
DB_PATH = "tkstore/example/Baseball.sqlite"
AGENT_SQL_PATH = "tkstore/example/execution_query.sql"
GOLD_SQL_PATH = "tkstore/example/gold.sql"
TMP_STORE_CSV = "tmp/quickstart_tkstore_example.csv"
DB_NAME = "Baseball"

for required in [EXAMPLE_JSON, DB_PATH, AGENT_SQL_PATH, GOLD_SQL_PATH]:
    if not Path(required).exists():
        raise FileNotFoundError(f"Missing required file: {required}")

print("\n[1/4] Initializing LLM provider …")
cfg = tkboost.init(provider="auto")
print(f"       Provider: {cfg['provider']}  |  Model: {cfg['model']}")

store_path = Path(TMP_STORE_CSV)
store_path.parent.mkdir(parents=True, exist_ok=True)
if store_path.exists():
    store_path.unlink()
print(f"       Quickstart store: {store_path}")

executor = SQLiteExecutor(DB_PATH)
draft_sql = Path(AGENT_SQL_PATH).read_text(encoding="utf-8")
gold_sql = Path(GOLD_SQL_PATH).read_text(encoding="utf-8")

print("\n[2/4] Generating tribal knowledge store …")
store = tkboost.generate(
    example_json=EXAMPLE_JSON,
    store=str(store_path),
    executor=executor,
    debug=True,
)
print(f"       Store written to: {store.path}")

print("\n[3/4] Refining existing draft SQL with tribal knowledge …")
result = tkboost.sql(
    draft=draft_sql,
    executor=executor,
    store=store,
    db_name=DB_NAME,
)

print("\n[4/4] Comparing results …\n")
_, agent_rows = executor.execute(draft_sql)
_, refined_rows = executor.execute(result["refined_sql"])
_, gold_rows = executor.execute(gold_sql)

print(f"agent:   {agent_rows[:3]}")
print(f"refined: {refined_rows[:3]}")
print(f"gold:    {gold_rows[:3]}")
