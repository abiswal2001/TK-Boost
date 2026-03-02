import json
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import psycopg  # type: ignore
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore

from .base import Executor


class PostgresExecutor(Executor):
    """Postgres executor.

    Accepts either:
    - a full DSN string (e.g., postgresql://user:pass@host:5432/dbname)
    - or a path to a JSON file containing connection kwargs for psycopg.connect
      (keys like host, port, user, password, dbname).
    """

    def __init__(self, dsn_or_cred: str):
        self.dsn_or_cred = dsn_or_cred

    def execute(self, sql: str) -> Tuple[Optional[List[str]], List[Tuple]]:
        if psycopg is None:
            raise RuntimeError("psycopg is not installed. Install it to use PostgresExecutor.")

        conn = None
        cur = None
        try:
            connect_arg = self.dsn_or_cred or ""
            p = Path(connect_arg)
            if p.exists() and p.suffix.lower() == ".json":
                cfg = json.loads(p.read_text(encoding="utf-8"))
                conn = psycopg.connect(**cfg)
            else:
                conn = psycopg.connect(connect_arg)
            cur = conn.cursor()
            cur.execute(sql)

            rows = cur.fetchall() if cur.description else []
            headers = [d.name for d in cur.description] if cur.description else None
            return headers, rows
        finally:
            try:
                if cur is not None:
                    cur.close()
            except Exception:
                pass
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

