from typing import Optional


def resolve_sqlite_db_path(instance_id: str, db_id: str) -> Optional[str]:
    """Reuse logic from src/agents/cte_refiner.get_database_path if available.

    Returns absolute or relative path to a .sqlite DB for a given instance.
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cte_refiner", "src/agents/cte_refiner.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
        return mod.get_database_path(instance_id, db_id)
    except Exception:
        return None




