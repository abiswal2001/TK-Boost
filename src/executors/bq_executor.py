from typing import List, Tuple, Optional

try:
    from google.cloud import bigquery  # type: ignore
except Exception:  # pragma: no cover
    bigquery = None  # type: ignore

import os

from .base import Executor


class BigQueryExecutor(Executor):
    def __init__(self, credential_path: str = "src/executors/bigquery_credential.json"):
        self.credential_path = credential_path
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if bigquery is None:
                raise RuntimeError("google-cloud-bigquery not installed. Install it to use BigQueryExecutor.")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credential_path
            self._client = bigquery.Client()

    def execute(self, sql: str) -> Tuple[Optional[List[str]], List[Tuple]]:
        self._ensure_client()
        query_job = self._client.query(sql)
        result = query_job.result()
        headers = [field.name for field in result.schema] if result.schema else None
        rows = [tuple(row.values()) for row in result]
        return headers, rows
