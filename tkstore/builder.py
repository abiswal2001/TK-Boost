import csv
import io
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import litellm

from . import config
from .examples import load_example
from .harness import generate_memory_diff_first_turn
from .rules import generate_rules_from_diff
from .tagger_index import MemoryIndex, generate_tagged_memories_json
from src.executors.factory import make_executor


def add_memory(
    instance_id: str,
    clean_summary: str,
    database_memories: List[str],
    generic_memories: List[str],
    out_dir: str,
    model: str = "azure/o4-mini",
    verbose: bool = True,
    multiturn: bool = True,
) -> Optional[Dict[str, Any]]:
    """Create tagged JSON for memories and append them to the tkstore index CSV.

    This function calls the LLM tagger and then uses `MemoryIndex.append_tagged`
    to persist the results. Returns the parsed tagged JSON (or None on failure).
    """
    tagged = generate_tagged_memories_json(
        instance_id=instance_id,
        user_query="",
        db_name=None,
        gold_sql="",
        agent_sql="",
        clean_summary=clean_summary,
        database_memories=database_memories,
        generic_memories=generic_memories,
        evidence=None,
        minimal_required_edits=None,
        model=model,
        verbose=verbose,
        multiturn=multiturn,
    )

    if tagged is None:
        return None

    # Backward compatibility: historically callers passed a directory path.
    out_path = Path(out_dir)
    if out_path.suffix.lower() != ".csv":
        out_path.mkdir(parents=True, exist_ok=True)
        index_path = config.MEMORY_INDEX_PATH
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        index_path = str(out_path)

    idx = MemoryIndex(index_path)
    idx.append_tagged(tagged, str(out_path.parent), str(instance_id), verbose=verbose)
    return tagged


class MemoryBuilder:
    def __init__(self, out_dir: str, model: str = "azure/o4-mini", verbose: bool = True):
        self.out_dir = out_dir
        self.model = model
        self.verbose = verbose

    def add(self, instance_id: str, clean_summary: str, database_memories: List[str], generic_memories: List[str]):
        return add_memory(instance_id, clean_summary, database_memories, generic_memories, self.out_dir, model=self.model, verbose=self.verbose)


def _default_index_for_engine(engine: str) -> str:
    eng = (engine or "").lower()
    if eng == "sqlite":
        return config.TKSTORE_SQLITE_PATH
    if eng in ("bq", "bigquery"):
        return config.TKSTORE_BQ_PATH
    if eng == "snowflake":
        return config.TKSTORE_SF_PATH
    # Fallback to legacy path for unknown engines.
    return config.MEMORY_INDEX_PATH


def _read_optional_text(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def _rows_to_csv_text(headers: Optional[List[str]], rows: List[Any]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    if headers:
        writer.writerow(headers)
    for r in rows or []:
        writer.writerow(list(r))
    return buf.getvalue()


def _execute_sql_to_csv(executor, sql_text: str) -> str:
    headers, rows = executor.execute(sql_text)
    return _rows_to_csv_text(headers, rows)


def _extract_clean_summary(diff_output: str) -> str:
    m = re.search(r"CLEAN_SUMMARY\s*:\s*(.*)", diff_output or "", flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_memories_from_rules(rules_text: str) -> Dict[str, List[str]]:
    q_mems: List[str] = []
    g_mems: List[str] = []
    if not rules_text:
        return {"database_memories": q_mems, "generic_memories": g_mems}

    mq = re.search(
        r"(?:DATABASE_MEMORIES|QUESTION_MEMORIES)\s*:?\s*(.*?)(?=\s*(?:GENERIC_MEMORIES|$))",
        rules_text,
        flags=re.S | re.IGNORECASE,
    )
    mg = re.search(
        r"GENERIC_MEMORIES\s*:?\s*(.*?)(?=\s*(?:DATABASE_MEMORIES|QUESTION_MEMORIES|CLEAN_SUMMARY|$))",
        rules_text,
        flags=re.S | re.IGNORECASE,
    )
    q_block = mq.group(1).strip() if mq else ""
    g_block = mg.group(1).strip() if mg else ""

    def _split_items(block: str) -> List[str]:
        out: List[str] = []
        for ln in block.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.upper() in {"DATABASE_MEMORIES", "GENERIC_MEMORIES", "QUESTION_MEMORIES"}:
                continue
            ln = re.sub(r"^[\d\)\-\•\*]\s*", "", ln)
            if ln:
                out.append(ln)
        return out

    q_mems = _split_items(q_block)
    g_mems = _split_items(g_block)
    return {"database_memories": q_mems, "generic_memories": g_mems}


def _extract_sql_from_text(content: str) -> str:
    if not content:
        return ""
    m = re.search(r"```sql\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"<solution>([\s\S]*?)</solution>", content, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return content.strip()


def _generate_agent_sql(question: str, engine: str, db_info: Optional[str], external_evidence: Optional[str], model: str) -> str:
    prompt = (
        "You are an expert SQL writer. Generate a single SQL query for the question.\n"
        "Return SQL only (prefer ```sql fenced block```). No explanation.\n"
    )
    payload = f"[QUESTION]\n{question.strip()}\n"
    if db_info:
        payload += "\n[DB_INFO]\n" + db_info.strip() + "\n"
    if external_evidence:
        payload += "\n[EXTERNAL_EVIDENCE]\n" + external_evidence.strip() + "\n"
    payload += f"\n[ENGINE]\n{engine}\n"

    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": "You produce executable SQL only."},
            {"role": "user", "content": prompt + "\n" + payload},
        ],
    )
    try:
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        content = getattr(resp.choices[0].message, "content", "")
    sql = _extract_sql_from_text(content or "")
    if not sql:
        raise RuntimeError("Failed to generate agent SQL from model output.")
    return sql


def _append_clean_summary_row(index_path: str, example_id: str, database_id: str, clean_summary: str) -> None:
    if not clean_summary:
        return
    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = ["mem_id", "instance_id", "db", "scope", "sql_operations", "table", "column", "data_type", "nulls", "rule"]

    existing_lines = []
    if p.exists():
        existing_lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    if not existing_lines:
        with open(p, "w", encoding="utf-8", newline="") as wf:
            writer = csv.writer(wf)
            writer.writerow(header)
        next_id = 0
    else:
        next_id = max(0, len(existing_lines) - 1)

    row = [next_id, example_id, database_id or "all", "question", "NA", "NA", "NA", "NA", "NA", clean_summary.replace("\n", " ")]
    with open(p, "a", encoding="utf-8", newline="") as af:
        writer = csv.writer(af)
        writer.writerow(row)


def build_knowledge_from_example(
    example_json_path: str,
    index_path: Optional[str] = None,
    model: str = "azure/o4-mini",
    draft_sql_model: Optional[str] = None,
    max_turns: int = 6,
    verbose: bool = True,
    hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Build or extend tkstore from a single portable example.json.

    Minimal required setup for users:
      - example.json with question inline
      - gold.sql
    Optional:
      - agent.sql (if absent, generated using draft_sql_model/model)
      - gold_result.csv / agent_result.csv (if absent, executed on the database)
      - db_info_path, external_evidence_path
    """
    ex = load_example(example_json_path)
    engine = ex["engine"]
    example_id = ex["example_id"]
    database_id = ex["database_id"]
    question = ex["question"]

    gold_sql = Path(ex["gold_sql_path"]).read_text(encoding="utf-8")
    db_info = _read_optional_text(ex.get("db_info_path"))
    external_evidence = _read_optional_text(ex.get("external_evidence_path"))

    credential_or_db_path = ex.get("db_path") or ex.get("credential_path")
    executor = make_executor(engine, credential_or_db_path)
    try:
        agent_sql = _read_optional_text(ex.get("agent_sql_path"))
        if not agent_sql:
            agent_sql = _generate_agent_sql(
                question=question,
                engine=engine,
                db_info=db_info,
                external_evidence=external_evidence,
                model=draft_sql_model or model,
            )

        gold_result = _read_optional_text(ex.get("gold_result_path"))
        if not gold_result:
            gold_result = _execute_sql_to_csv(executor, gold_sql)

        agent_result = _read_optional_text(ex.get("agent_result_path"))
        if not agent_result:
            agent_result = _execute_sql_to_csv(executor, agent_sql)
    finally:
        close_fn = getattr(executor, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    prompt_context_parts: List[str] = []
    if db_info:
        prompt_context_parts.append("[DB_INFO]\n" + db_info.strip())
    if external_evidence:
        prompt_context_parts.append("[EXTERNAL_EVIDENCE]\n" + external_evidence.strip())
    prompt_context = "\n\n".join(prompt_context_parts) if prompt_context_parts else None

    diff_output = generate_memory_diff_first_turn(
        instance_id=example_id,
        user_query=question,
        agent_cte_text="",
        gold_sql_text=gold_sql,
        gold_result_csv_text=gold_result,
        agent_full_sql_text=agent_sql,
        agent_result_csv_text=agent_result,
        processed_trace_text=None,
        engine=engine,
        db_path_or_cred=credential_or_db_path,
        db_name=database_id,
        external_knowledge=prompt_context,
        max_turns=max_turns,
        model=model,
        verbose=verbose,
        hint=hint,
    ) or ""

    rules = generate_rules_from_diff(diff_output, agent_sql, agent_result, model=model, verbose=verbose) or ""
    clean_summary = _extract_clean_summary(diff_output)
    parsed = _extract_memories_from_rules(rules)

    tagged = generate_tagged_memories_json(
        instance_id=example_id,
        user_query=question,
        db_name=database_id,
        gold_sql=gold_sql,
        agent_sql=agent_sql,
        clean_summary=clean_summary,
        database_memories=parsed["database_memories"],
        generic_memories=parsed["generic_memories"],
        evidence=(diff_output[:2000] if diff_output else None),
        minimal_required_edits=None,
        model=model,
        verbose=verbose,
        multiturn=True,
    )
    if tagged is None:
        raise RuntimeError(f"Tagger failed for example {example_id}")

    target_index = index_path or _default_index_for_engine(engine)
    mem_index = MemoryIndex(target_index)
    Path(target_index).parent.mkdir(parents=True, exist_ok=True)
    mem_index.append_tagged(tagged, str(Path(target_index).parent), example_id, verbose=verbose)
    _append_clean_summary_row(target_index, example_id, database_id, clean_summary)

    return {
        "example_id": example_id,
        "database_id": database_id,
        "engine": engine,
        "index_path": target_index,
        "clean_summary": clean_summary,
        "database_memories_count": len(parsed["database_memories"]),
        "generic_memories_count": len(parsed["generic_memories"]),
    }


def build_knowledge_from_examples_dir(
    examples_root: str,
    index_path: Optional[str] = None,
    model: str = "azure/o4-mini",
    draft_sql_model: Optional[str] = None,
    max_turns: int = 6,
    verbose: bool = True,
    hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build or extend tkstore from all example.json files under a directory."""
    root = Path(examples_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"examples_root not found: {root}")

    results: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("example.json")):
        try:
            results.append(
                build_knowledge_from_example(
                    example_json_path=str(p),
                    index_path=index_path,
                    model=model,
                    draft_sql_model=draft_sql_model,
                    max_turns=max_turns,
                    verbose=verbose,
                    hint=hint,
                )
            )
        except Exception as e:
            results.append({"example_json": str(p), "error": str(e)})
    return results


