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

    def execute(self, sql: str) -> Tuple[Optional[List[str]], List[Tuple]]:
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not installed. Install it to use BigQueryExecutor.")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credential_path
        client = bigquery.Client()
        query_job = client.query(sql)
        result = query_job.result()
        headers = [field.name for field in result.schema] if result.schema else None
        rows = [tuple(row.values()) for row in result]
        return headers, rows

from typing import List, Tuple, Optional

try:
    from google.cloud import bigquery  # type: ignore
except Exception:  # pragma: no cover
    bigquery = None  # type: ignore

import json
import os

from .base import Executor


class BigQueryExecutor(Executor):
    def __init__(self, credential_path: str = "bigquery_credential.json"):
        self.credential_path = credential_path

    def execute(self, sql: str) -> Tuple[Optional[List[str]], List[Tuple]]:
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not installed. Install it to use BigQueryExecutor.")
        # Set credentials for the client
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credential_path
        client = bigquery.Client()
        query_job = client.query(sql)
        result = query_job.result()
        headers = [field.name for field in result.schema] if result.schema else None
        rows = [tuple(row.values()) for row in result]
        return headers, rows


