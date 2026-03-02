import json
from pathlib import Path
from typing import Dict, Any, Optional


REQUIRED_FIELDS = ("example_id", "database_id", "engine", "question", "gold_sql_path")
OPTIONAL_PATH_FIELDS = (
    "agent_sql_path",
    "gold_result_path",
    "agent_result_path",
    "db_info_path",
    "external_evidence_path",
    "credential_path",
)


def load_example(example_json_path: str) -> Dict[str, Any]:
    """Load and validate a training example from a single JSON file.

    Required fields:
      - example_id
      - database_id
      - engine
      - question
      - gold_sql_path

    Notes:
      - For sqlite examples, db_path is required.
      - Relative paths are resolved against the example JSON parent directory.
    """
    p = Path(example_json_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"example.json not found: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_FIELDS if not raw.get(k)]
    if missing:
        raise ValueError(f"Missing required fields in {p}: {missing}")

    engine = str(raw["engine"]).lower().strip()
    base_dir = p.parent

    ex: Dict[str, Any] = {
        "example_id": str(raw["example_id"]).strip(),
        "database_id": str(raw["database_id"]).strip(),
        "engine": engine,
        "question": str(raw["question"]).strip(),
        "db_path": None,
        "gold_sql_path": None,
    }

    if engine == "sqlite":
        db_path = raw.get("db_path")
        if not db_path:
            raise ValueError(f"sqlite example requires db_path: {p}")
        db_path_resolved = Path(db_path)
        if not db_path_resolved.is_absolute():
            db_path_resolved = (base_dir / db_path_resolved).resolve()
        if not db_path_resolved.exists():
            raise FileNotFoundError(f"db_path does not exist: {db_path_resolved}")
        ex["db_path"] = str(db_path_resolved)
    else:
        # Non-sqlite engines can use credential_path and/or external connector defaults.
        ex["db_path"] = None

    gold_sql_path = Path(raw["gold_sql_path"])
    if not gold_sql_path.is_absolute():
        gold_sql_path = (base_dir / gold_sql_path).resolve()
    if not gold_sql_path.exists():
        raise FileNotFoundError(f"gold_sql_path does not exist: {gold_sql_path}")
    ex["gold_sql_path"] = str(gold_sql_path)

    for key in OPTIONAL_PATH_FIELDS:
        val = raw.get(key)
        if not val:
            ex[key] = None
            continue
        resolved = Path(val)
        if not resolved.is_absolute():
            resolved = (base_dir / resolved).resolve()
        ex[key] = str(resolved)

    return ex


def scaffold_example_dir(target_dir: str, example_id: str = "example_001") -> str:
    """Create a minimal example directory scaffold for users."""
    out = Path(target_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    template = {
        "example_id": example_id,
        "database_id": "my_database",
        "engine": "sqlite",
        "question": "Write your natural language question here.",
        "db_path": "/absolute/path/to/database.sqlite",
        "gold_sql_path": "gold.sql",
        "agent_sql_path": None,
        "gold_result_path": None,
        "agent_result_path": None,
        "db_info_path": None,
        "external_evidence_path": None,
        "credential_path": None,
    }

    json_path = out / "example.json"
    if not json_path.exists():
        json_path.write_text(json.dumps(template, indent=2), encoding="utf-8")

    gold_sql = out / "gold.sql"
    if not gold_sql.exists():
        gold_sql.write_text("-- Put the gold SQL here.\n", encoding="utf-8")

    return str(json_path)
