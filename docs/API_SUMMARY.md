# TK-Boost API Summary

This document summarizes the current high-level `tkboost` SDK APIs.

## Imports

```python
import tkboost
from tkboost import (
    Executor,
    SQLiteExecutor,
    BigQueryExecutor,
    SnowflakeExecutor,
    PostgresExecutor,
    TKStore,
    TKStoreEntry,
)
```

## 1) `tkboost.init(...)`

Initialize LLM provider/model configuration used by `generate(...)` and `sql(...)`.

```python
tkboost.init(
    provider="auto",          # "auto" | "openai" | "azure"
    api_key=None,
    base_url=None,
    api_version=None,
    model=None,
    draft_sql_model=None,
    azure_api_key=None,
    azure_base_url=None,
)
```

### Provider behavior
- `provider="auto"`:
  - Uses OpenAI if `OPENAI_API_KEY` (or `api_key`) is available.
  - Else uses Azure if `AZURE_API_KEY` / `AZURE_OPENAI_API_KEY` (or `azure_api_key`) is available.
  - Else defaults to OpenAI mode.
- OpenAI defaults model to `gpt-4o-mini`.
- Azure defaults model to `azure/o4-mini`.

### Common env vars used
- OpenAI: `OPENAI_API_KEY`, optional `OPENAI_API_BASE`
- Azure: `AZURE_API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_API_BASE`, `AZURE_OPENAI_ENDPOINT`, `AZURE_API_VERSION`

## 2) Executors

All executors implement:

```python
class Executor:
    def execute(self, sql: str) -> tuple[list[str] | None, list[tuple]]:
        ...
```

### `SQLiteExecutor`
```python
executor = SQLiteExecutor(db_path="/abs/path/to/db.sqlite")
```

### `BigQueryExecutor`
```python
executor = BigQueryExecutor(credential_path="/abs/path/to/service_account.json")
```

### `SnowflakeExecutor`
```python
executor = SnowflakeExecutor(
    credential_path="/abs/path/to/snowflake_credential.json",
    fix_quote_mode=False,
)
```

### `PostgresExecutor`
```python
# Either DSN:
executor = PostgresExecutor("postgresql://user:pass@host:5432/dbname")

# Or JSON credential file path with psycopg connect kwargs:
executor = PostgresExecutor("/abs/path/to/postgres_credentials.json")
```

## 3) `tkboost.generate(...)`

Generate tribal knowledge from one example or a directory of examples, and persist into a store CSV.

```python
store_obj = tkboost.generate(
    example_json=None,        # path to a single example.json
    examples_dir=None,        # path to directory containing example folders
    store=None,               # path to tkstore csv
    executor=None,            # optional Executor instance
    model=None,
    draft_sql_model=None,
    max_turns=6,
    verbose=True,
    hint=None,
    debug=False,              # writes detailed traces/artifacts when True
)
```

### Notes
- Provide exactly one of `example_json` or `examples_dir`.
- Returns a `TKStore` object.
- If `store` does not exist, it is created (with canonical header).
- `examples_dir` processes all example folders under that directory.
- `example_json` is useful for single-example runs; most users should use `examples_dir`.
- If `debug=True`, per-example debug artifacts are written (including `llm_interactions.json` and SQL/rule artifacts).

## 4) `tkboost.sql(...)`

Refine a draft SQL query using tribal knowledge rules from a store.

```python
result = tkboost.sql(
    question=None,            # required only if draft is omitted
    draft=None,               # agent draft sql
    executor=None,            # optional Executor; if passed, refined SQL is executed
    store=None,               # tkstore path OR TKStore object
    model=None,
    db_name=None,             # retrieval db filter
    db_info=None,             # optional schema/context text used for draft generation
    use_llm_filtering=False,
)
```

### Behavior
- If `draft` is missing, TK-Boost generates a draft from `question` using configured LLM credentials.
- Retrieves matching rules from `store`.
- Produces refined SQL via LLM.
- If `executor` is provided, executes the refined SQL and returns a preview.

### Return shape
- `draft_sql`
- `refined_sql`
- `rule_count`
- `rules_used` (up to 40)
- `execution`:
  - `ok` (`True` / `False` / `None`)
  - `error`
  - `preview_headers`
  - `preview_rows` (up to 10 rows)

## 5) `TKStore` object

`TKStore` is a CSV-backed store wrapper.

```python
store = TKStore("/abs/path/to/tkstore.csv")
```

### Core methods
- `store.path` -> bound csv path
- `store.exists()` -> bool
- `store.rows()` -> list of csv row dicts
- `store.retriever()` -> `MemoryRetriever`
- `store.retrieve(sql_text, generic_only=False, use_llm_filtering=False, llm_model=None, db_name=None)`
- `store.visualize(port=8501, open_browser=True)` -> launches dashboard

### CRUD methods
- `store.insert(entry: TKStoreEntry) -> TKStoreEntry`
- `store.insert_many(entries: list[TKStoreEntry]) -> list[TKStoreEntry]`
- `store.update(mem_id: int, **fields: str) -> bool`
- `store.delete(mem_id: int) -> bool` (reindexes `mem_id` to keep contiguous ids)

## 6) `TKStoreEntry`

Single row model for store inserts/updates.

```python
entry = TKStoreEntry(
    mem_id=None,
    instance_id="local007",
    db="Baseball",
    scope="db",                 # or generic/question
    sql_operations="strftime;cast;round",
    table="player",
    column="player.final_game;player.debut",
    data_type="date",
    nulls="all",
    rule="ENSURE: ...",
)
store.insert(entry)
```

## 7) Training Example JSON (for `generate`)

Required fields in `example.json`:
- `example_id`
- `engine`
- `question`
- `gold_sql_path`

Common optional fields:
- `db_name` (preferred; legacy `database_id` also accepted)
- `db_path` (needed for sqlite if no executor is passed)
- `agent_sql_path`
- `gold_result_path`
- `agent_result_path`
- `db_info_path`
- `external_evidence_path`
- `credential_path`

## 8) Minimal End-to-End Example

```python
import tkboost
from tkboost import SQLiteExecutor

tkboost.init(provider="auto", model="gpt-4o-mini")

executor = SQLiteExecutor("tkstore/example/Baseball.sqlite")

store = tkboost.generate(
    example_json="tkstore/example/example.json",  # local007
    store="tkstore/tkstore_example.csv",
    executor=executor,
    debug=True,
)

result = tkboost.sql(
    question="Compute the average career span in years for baseball players.",
    executor=executor,
    store=store,
    db_name="Baseball",
)

print(result["refined_sql"])
store.visualize(port=8501)
```
