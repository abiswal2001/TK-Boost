from typing import Optional

from src.executors import SQLiteExecutor, SnowflakeExecutor, BigQueryExecutor
from src.utils.agent_utils import infer_engine
from src.utils.db_paths import resolve_sqlite_db_path


def make_executor(engine: str, db_path_or_cred: Optional[str]):
    eng = (engine or "sqlite").lower()
    if eng == "sqlite":
        if not db_path_or_cred:
            raise ValueError("SQLite requires a database path")
        return SQLiteExecutor(db_path_or_cred)
    if eng == "snowflake":
        return SnowflakeExecutor(db_path_or_cred or "src/executors/snowflake_credential.json")
    if eng in ("bq", "bigquery"):
        return BigQueryExecutor(db_path_or_cred or "src/executors/bigquery_credential.json")
    raise ValueError(f"Unsupported engine: {engine}")


def make_executor_for_instance(instance_id: str, db_id: str):
    engine = infer_engine(instance_id)
    if engine == "sqlite":
        db_path = resolve_sqlite_db_path(instance_id, db_id)
        if not db_path:
            raise FileNotFoundError(f"Could not resolve SQLite DB for {instance_id} ({db_id})")
        return SQLiteExecutor(db_path)
    if engine == "snowflake":
        return SnowflakeExecutor("src/executors/snowflake_credential.json")
    if engine in ("bq", "bigquery"):
        return BigQueryExecutor("src/executors/bigquery_credential.json")
    raise ValueError(f"Unsupported engine: {engine}")




