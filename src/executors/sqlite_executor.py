import sqlite3
import threading
from typing import List, Tuple, Optional

from .base import Executor


class SQLiteExecutor(Executor):
    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute(self, sql: str) -> Tuple[Optional[List[str]], List[Tuple]]:
        conn = sqlite3.connect(self.db_path)
        timeout_occurred = threading.Event()
        
        def timeout_handler():
            timeout_occurred.set()
            conn.interrupt()
        
        timer = threading.Timer(120.0, timeout_handler)
        timer.start()
        
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            headers = [d[0] for d in cur.description] if cur.description else None
            timer.cancel()
            
            if timeout_occurred.is_set():
                raise TimeoutError("SQL query execution exceeded 120 seconds")
            
            return headers, rows
        except sqlite3.OperationalError as e:
            timer.cancel()
            if timeout_occurred.is_set():
                raise TimeoutError("SQL query execution exceeded 120 seconds") from e
            raise
        finally:
            timer.cancel()
            try:
                cur.close()
            except Exception:
                pass
            conn.close()


