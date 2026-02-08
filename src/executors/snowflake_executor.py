from typing import List, Tuple, Optional

try:
    import snowflake.connector  # type: ignore
except Exception:  # pragma: no cover
    snowflake = None  # type: ignore

import json
import re

try:
    from sqlglot import parse_one, exp  # type: ignore
except Exception:  # pragma: no cover
    parse_one = None  # type: ignore
    exp = None  # type: ignore

from .base import Executor


class SnowflakeExecutor(Executor):
    """Snowflake executor with persistent connection support.
    
    The connection is established once on first execute() call and reused
    for subsequent queries. Call close() explicitly when done (e.g., at end
    of agent run) to clean up the connection.
    """
    def __init__(self, credential_path: str = "src/executors/snowflake_credential.json", fix_quote_mode: bool = False):
        self.credential_path = credential_path
        self.fix_quote_mode = fix_quote_mode
        self._conn = None
        self._cursor = None

    def _ensure_connection(self):
        """Establish connection if not already connected."""
        if self._conn is None:
            if snowflake is None:
                raise RuntimeError("snowflake-connector-python not installed. Install it to use SnowflakeExecutor.")
            creds = json.load(open(self.credential_path))
            self._conn = snowflake.connector.connect(**creds)
            self._cursor = self._conn.cursor()

    def execute(self, sql: str) -> Tuple[Optional[List[str]], List[Tuple]]:
        """Execute SQL query using persistent connection."""
        self._ensure_connection()
        try:
            current_sql = sql
            if self.fix_quote_mode:
                try:
                    current_sql = self._quote_columns(current_sql)
                except Exception:
                    pass
            self._cursor.execute(current_sql)
            rows = self._cursor.fetchall()
            headers = [d[0] for d in self._cursor.description] if self._cursor.description else None
            return headers, rows
        except Exception as e:
            # On error, close connection so next execute() will reconnect
            self.close()
            raise e

    def close(self):
        """Explicitly close the connection and cursor."""
        if self._cursor:
            try:
                self._cursor.close()
            except Exception:
                pass
            self._cursor = None
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __del__(self):
        """Cleanup connection on garbage collection."""
        self.close()

    def _quote_columns(self, query: str) -> str:
        """Quote unquoted column identifiers using sqlglot parsing.

        Best-effort: if sqlglot is unavailable or parsing fails, returns the original query.
        """
        if parse_one is None or exp is None:
            return query
        try:
            parsed = parse_one(query, dialect="snowflake")
            col_names = [col.text("this") for col in parsed.find_all(exp.Column)]
            col_names = list(set(col_names))
            for col_name in col_names:
                quoted_pattern = r'\b"' + re.escape(col_name) + r'"\b'
                if not re.search(quoted_pattern, query):
                    pattern = r'\b' + re.escape(col_name) + r'\b'
                    query = re.sub(pattern, f'"{col_name}"', query)
            return query
        except Exception:
            return query
