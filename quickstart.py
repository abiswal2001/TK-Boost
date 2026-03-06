"""
TK-Boost Quick Start
====================
End-to-end example: draft SQL -> generate tribal knowledge -> refine SQL.

Prerequisites (run once):
  1. pip install -r requirements.txt
  2. Set LLM env vars (OPENAI_API_KEY  *or*  AZURE_API_KEY + AZURE_API_BASE + AZURE_API_VERSION)
  3. Download Baseball.sqlite into tkstore/example/  (see README for instructions)

Usage:
  python quickstart.py
"""

from pathlib import Path

import tkboost
from tkboost import SQLiteExecutor, SQLAgent

DB_PATH = "tkstore/example/Baseball.sqlite"
EXAMPLE_JSON = "tkstore/example/example.json"
STORE_CSV = "tkstore/tkstore_example.csv"
GOLD_SQL_PATH = "tkstore/example/gold.sql"

# ── Preflight ────────────────────────────────────────────────────────────────

if not Path(DB_PATH).exists():
    raise FileNotFoundError(
        f"{DB_PATH} not found. Download it first — see README step 3."
    )

# ── 1. Initialize ────────────────────────────────────────────────────────────

print("\n[1/5] Initializing LLM provider …")
cfg = tkboost.init(provider="auto")
print(f"       Provider: {cfg['provider']}  |  Model: {cfg['model']}")

# ── 2. Create executor + agent ───────────────────────────────────────────────

executor = SQLiteExecutor(DB_PATH)
agent = SQLAgent()

# ── 3. Translate question → draft SQL ────────────────────────────────────────

question = "Compute the average career span in years for baseball players."

print(f"\n[2/5] Translating question → draft SQL …")
draft = agent.translate(
    question=question,
    executor=executor,
    db_name="Baseball",
)
print(f"       Draft SQL (first 120 chars): {draft['sql'][:120]}")

# ── 4. Generate tribal knowledge ─────────────────────────────────────────────

print(f"\n[3/5] Generating tribal knowledge store …")
store = tkboost.generate(
    example_json=EXAMPLE_JSON,
    store=STORE_CSV,
    executor=executor,
    debug=True,
)
print(f"       Store written to: {store.path}")

# ── 5. Refine draft using tribal knowledge ───────────────────────────────────

print(f"\n[4/5] Refining draft SQL with tribal knowledge …")
result = tkboost.sql(
    draft=draft["sql"],
    executor=executor,
    store=store,
    db_name="Baseball",
)
print(f"       Refined SQL (first 120 chars): {result['refined_sql'][:120]}")

# ── 6. Compare results ──────────────────────────────────────────────────────

print(f"\n[5/5] Comparing results …\n")

gold_sql = Path(GOLD_SQL_PATH).read_text(encoding="utf-8")

_, agent_rows = executor.execute(draft["sql"])
_, refined_rows = executor.execute(result["refined_sql"])
_, gold_rows = executor.execute(gold_sql)

print(f"  Agent (draft):    {agent_rows[:3]}")
print(f"  Refined (TK):     {refined_rows[:3]}")
print(f"  Gold (expected):  {gold_rows[:3]}")
print()

if refined_rows and gold_rows:
    match = str(refined_rows[0]) == str(gold_rows[0])
    print(f"  {'✓ Refined matches gold!' if match else '✗ Refined differs from gold.'}")

print("\nDone.")
