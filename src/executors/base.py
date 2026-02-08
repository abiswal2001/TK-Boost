from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class Executor(ABC):
    @abstractmethod
    def execute(self, sql: str) -> Tuple[Optional[List[str]], List[Tuple]]:
        raise NotImplementedError

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class Executor(ABC):
    """Abstract base for SQL execution across engines.

    Implementations must execute a single SQL statement and return (headers, rows).
    """

    @abstractmethod
    def execute(self, sql: str) -> Tuple[Optional[List[str]], List[Tuple]]:
        """Execute a single SQL statement.

        Returns:
            headers: list of column names (or None if unknown)
            rows: list of tuples
        """
        raise NotImplementedError


