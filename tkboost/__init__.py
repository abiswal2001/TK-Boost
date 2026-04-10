"""Public high-level API for TK-Boost quickstart usage."""

import os
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.executors.base import Executor
from src.executors.sqlite_executor import SQLiteExecutor
from src.executors.bq_executor import BigQueryExecutor
from src.executors.snowflake_executor import SnowflakeExecutor
from src.executors.postgres_executor import PostgresExecutor
from tkstore.builder import (
    build_knowledge_from_example,
    build_knowledge_from_examples_dir,
)
from tkstore.tagger_index import MemoryRetriever


_STATE: Dict[str, Any] = {
    "provider": "auto",
    "model": "azure/gpt-5",
    "draft_sql_model": None,
}


@dataclass
class TKStoreEntry:
    """Single tkstore row."""

    mem_id: Optional[int] = None
    instance_id: str = ""
    db: str = "all"
    scope: str = "generic"
    sql_operations: str = "all"
    table: str = "all"
    column: str = "all"
    data_type: str = "all"
    nulls: str = "all"
    rule: str = ""

    def to_row(self) -> Dict[str, str]:
        return {
            "mem_id": "" if self.mem_id is None else str(self.mem_id),
            "instance_id": self.instance_id,
            "db": self.db,
            "scope": self.scope,
            "sql_operations": self.sql_operations,
            "table": self.table,
            "column": self.column,
            "data_type": self.data_type,
            "nulls": self.nulls,
            "rule": self.rule,
        }


class TKStore:
    """Light wrapper over a tkstore CSV path."""

    HEADER = [
        "mem_id",
        "instance_id",
        "db",
        "scope",
        "sql_operations",
        "table",
        "column",
        "data_type",
        "nulls",
        "rule",
    ]

    def __init__(self, store: Optional[str], last_generate_result: Any = None):
        self.store = store
        self.last_generate_result = last_generate_result
        if self.store:
            self._ensure_well_formed()

    @property
    def path(self) -> Optional[str]:
        return self.store

    def exists(self) -> bool:
        return bool(self.store) and Path(self.store).exists()

    def retriever(self) -> MemoryRetriever:
        if not self.store:
            raise ValueError("TKStore has no bound store path.")
        return MemoryRetriever(self.store)

    def retrieve(
        self,
        sql_text: str,
        generic_only: bool = False,
        use_llm_filtering: bool = False,
        llm_model: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.store:
            raise ValueError("TKStore has no bound store path.")
        model = llm_model or _STATE.get("model") or "azure/gpt-5"
        return MemoryRetriever(self.store).retrieve(
            sql_text=sql_text,
            generic_only=generic_only,
            use_llm_filtering=use_llm_filtering,
            llm_model=model,
            db=db_name,
        )

    def rows(self) -> List[Dict[str, Any]]:
        if not self.store:
            return []
        p = Path(self.store)
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def insert(self, entry: TKStoreEntry) -> TKStoreEntry:
        if not self.store:
            raise ValueError("TKStore has no bound store path.")
        rows = self.rows()
        next_id = 0
        for r in rows:
            try:
                next_id = max(next_id, int(r.get("mem_id", "-1")) + 1)
            except Exception:
                pass
        if entry.mem_id is None:
            entry.mem_id = next_id
        rows.append(entry.to_row())
        self._write_rows(rows)
        return entry

    def insert_many(self, entries: List[TKStoreEntry]) -> List[TKStoreEntry]:
        out: List[TKStoreEntry] = []
        for e in entries:
            out.append(self.insert(e))
        return out

    def update(self, mem_id: int, **fields: str) -> bool:
        if not self.store:
            raise ValueError("TKStore has no bound store path.")
        rows = self.rows()
        updated = False
        allowed = set(self.HEADER) - {"mem_id"}
        for r in rows:
            try:
                if int(r.get("mem_id", "-1")) != int(mem_id):
                    continue
            except Exception:
                continue
            for k, v in fields.items():
                if k in allowed:
                    r[k] = str(v)
            updated = True
            break
        if updated:
            self._write_rows(rows)
        return updated

    def delete(self, mem_id: int) -> bool:
        if not self.store:
            raise ValueError("TKStore has no bound store path.")
        rows = self.rows()
        kept: List[Dict[str, Any]] = []
        removed = False
        for r in rows:
            try:
                if int(r.get("mem_id", "-1")) == int(mem_id):
                    removed = True
                    continue
            except Exception:
                pass
            kept.append(r)
        if removed:
            # Reassign mem_id to keep contiguous indexing.
            for i, r in enumerate(kept):
                r["mem_id"] = str(i)
            self._write_rows(kept)
        return removed

    def visualize(self, port: int = 8501, open_browser: bool = True) -> None:
        """Launch a localhost dashboard to explore the store and its debug traces."""
        if not self.store:
            raise ValueError("TKStore has no bound store path.")
        from tkboost.dashboard import serve
        serve(store_path=self.store, port=port, open_browser=open_browser)

    def __repr__(self) -> str:
        return f"TKStore(store={self.store!r})"

    def _ensure_well_formed(self) -> None:
        if not self.store:
            return
        p = Path(self.store)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with open(p, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADER)
                writer.writeheader()
            return
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
        if cols != self.HEADER:
            raise ValueError(
                f"Store CSV is not well formed. Expected columns: {self.HEADER}, got: {cols}"
            )

    def _write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not self.store:
            return
        self._ensure_well_formed()
        with open(self.store, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: str(r.get(k, "")) for k in self.HEADER})


def init(
    provider: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    model: Optional[str] = None,
    draft_sql_model: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize LLM auth/config for TK-Boost.

    Supports provider:
      - "auto" (default): choose OpenAI if OPENAI_API_KEY is set, else Azure if Azure keys are set
      - "openai"
      - "azure"
    """
    provider = (provider or "auto").lower().strip()
    if provider not in {"auto", "openai", "azure"}:
        raise ValueError("provider must be one of: auto, openai, azure")

    selected = provider
    if selected == "auto":
        if os.environ.get("OPENAI_API_KEY") or api_key:
            selected = "openai"
        elif os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or azure_api_key:
            selected = "azure"
        else:
            selected = "openai"

    if selected == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_API_BASE"] = base_url
        default_model = "gpt-5"
    else:
        if azure_api_key:
            os.environ["AZURE_API_KEY"] = azure_api_key
            os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
        elif api_key:
            os.environ["AZURE_API_KEY"] = api_key
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
        if azure_base_url:
            os.environ["AZURE_API_BASE"] = azure_base_url
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_base_url
        elif base_url:
            os.environ["AZURE_API_BASE"] = base_url
            os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
        if api_version:
            os.environ["AZURE_API_VERSION"] = api_version
        default_model = "azure/gpt-5"

    _STATE["provider"] = selected
    _STATE["model"] = model or default_model
    _STATE["draft_sql_model"] = draft_sql_model

    return {
        "provider": _STATE["provider"],
        "model": _STATE["model"],
        "draft_sql_model": _STATE["draft_sql_model"],
    }


def generate(
    example_json: Optional[str] = None,
    examples_dir: Optional[str] = None,
    store: Optional[str] = None,
    executor: Optional[Executor] = None,
    model: Optional[str] = None,
    draft_sql_model: Optional[str] = None,
    max_turns: int = 6,
    verbose: bool = True,
    hint: Optional[str] = None,
    debug: bool = False,
    sim_gate: bool = False,
) -> TKStore:
    """Generate tribal knowledge and write/append to a store CSV.

    - If store exists, rows are appended.
    - If store does not exist, it is created.
    - Provide exactly one of: example_json or examples_dir.
    - Returns a TKStore wrapper for the target CSV.
    - If debug=True, writes per-example debug artifacts (LLM traces, intermediate SQL/results).
    - sim_gate defaults to False; set True to enable similarity-based rejection of overly gold-like SQL edits during repair.
    """
    if bool(example_json) == bool(examples_dir):
        raise ValueError("Provide exactly one of example_json or examples_dir")

    effective_model = model or _STATE.get("model") or "azure/gpt-5"
    effective_draft = draft_sql_model if draft_sql_model is not None else _STATE.get("draft_sql_model")
    target_store = store
    tk_store = TKStore(target_store) if target_store else None

    if example_json:
        result = build_knowledge_from_example(
            example_json_path=example_json,
            store=target_store,
            tkstore_obj=tk_store,
            executor=executor,
            model=effective_model,
            draft_sql_model=effective_draft,
            max_turns=max_turns,
            verbose=verbose,
            hint=hint,
            debug=debug,
            sim_gate=sim_gate,
        )
        return TKStore(store=result.get("store") or target_store, last_generate_result=result)

    batch_result = build_knowledge_from_examples_dir(
        examples_root=examples_dir or "",
        store=target_store,
        tkstore_obj=tk_store,
        executor=executor,
        model=effective_model,
        draft_sql_model=effective_draft,
        max_turns=max_turns,
        verbose=verbose,
        hint=hint,
        debug=debug,
        sim_gate=sim_gate,
    )
    inferred_store = target_store
    if not inferred_store:
        for item in batch_result:
            st = item.get("store") if isinstance(item, dict) else None
            if st:
                inferred_store = st
                break
    return TKStore(store=inferred_store, last_generate_result=batch_result)


def _extract_sql_from_text(content: str) -> str:
    if not content:
        return ""
    m = re.search(r"```sql\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return content.strip()


def _infer_engine_from_executor(executor: Optional[Executor]) -> str:
    if executor is None:
        return "sqlite"
    name = executor.__class__.__name__.lower()
    if "snowflake" in name:
        return "snowflake"
    if "bigquery" in name or "bq" in name:
        return "bigquery"
    if "postgres" in name:
        return "postgres"
    return "sqlite"


def _infer_executor_target(executor: Executor) -> str:
    """Infer db/credential target path from a concrete executor object."""
    for attr in ("db_path", "credential_path", "dsn_or_cred"):
        val = getattr(executor, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    raise ValueError(
        "Unable to infer executor target path/credential from executor. "
        "Expected one of attributes: db_path, credential_path, dsn_or_cred."
    )


class SQLAgent:
    """Thin ReAct SQL agent wrapper around src.agents.sql_agent_runner.run_agent.

    This wrapper uses tkboost.init()-configured credentials/model defaults and exposes
    a compact interface for generating an executable draft SQL.
    """

    def __init__(self, model: Optional[str] = None, max_turns: int = 25,
                 verbose: bool = False, trace_dir: Optional[str] = "traces"):
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose
        self.trace_dir = trace_dir

    def translate(
        self,
        question: str,
        executor: Executor,
        db_name: str = "default_db",
        instance_id: str = "adhoc",
        predicted_cte_hint: Optional[str] = None,
        predicted_schema_hint: Optional[str] = None,
        schema_context: Optional[str] = None,
        external_knowledge: Optional[str] = None,
        expected_output_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("question is required")
        if executor is None:
            raise ValueError("executor is required")

        from src.agents.sql_agent_runner import Instance, run_agent, generate_processed_trace  # lazy import
        import json
        import time

        model = self.model or _STATE.get("model") or "azure/gpt-5"
        engine = _infer_engine_from_executor(executor)
        db_path_or_cred = _infer_executor_target(executor)
        inst = Instance(
            instance_id=instance_id,
            db=db_name,
            question=question,
            external_knowledge=external_knowledge,
        )

        # Compute trace base path so sub-agents can save traces alongside the main agent
        trace_base_path = None
        if self.trace_dir:
            ts = time.strftime("%Y%m%d_%H%M%S")
            trace_base_path = str(Path(self.trace_dir) / f"{instance_id}_{ts}")

        final_sql, headers, rows, messages, exec_used = run_agent(
            inst=inst,
            engine=engine,
            db_path_or_cred=db_path_or_cred,
            model=model,
            predicted_cte_hint=predicted_cte_hint,
            predicted_schema_hint=predicted_schema_hint,
            schema_context=schema_context,
            external_knowledge=external_knowledge,
            expected_output_format=expected_output_format,
            max_turns=self.max_turns,
            train_context_file=None,
            verbose=self.verbose,
            trace_dir=trace_base_path,
        )
        try:
            if hasattr(exec_used, "close"):
                exec_used.close()
        except Exception:
            pass

        # Save traces (sub-agent traces already saved by run_agent via trace_dir)
        trace_path = None
        if trace_base_path:
            trace_base = Path(trace_base_path)
            trace_base.mkdir(parents=True, exist_ok=True)

            # messages.json — full LLM conversation
            (trace_base / "messages.json").write_text(
                json.dumps(messages, indent=2), encoding="utf-8"
            )
            # processed_trace.txt — human-readable trace
            (trace_base / "processed_trace.txt").write_text(
                generate_processed_trace(messages), encoding="utf-8"
            )
            # execution_query.sql — final SQL
            (trace_base / "execution_query.sql").write_text(
                final_sql or "", encoding="utf-8"
            )
            # execution_result.csv — query results
            if headers and rows:
                import csv as _csv
                with open(trace_base / "execution_result.csv", "w", newline="", encoding="utf-8") as f:
                    writer = _csv.writer(f)
                    writer.writerow(headers)
                    for r in rows:
                        writer.writerow(list(r))

            trace_path = str(trace_base)
            if self.verbose:
                print(f"\n📁 Traces saved to: {trace_path}")

        return {
            "sql": final_sql,
            "headers": headers,
            "row_count": len(rows or []),
            "preview_rows": [list(r) for r in (rows or [])[:10]],
            "messages": messages,
            "model": model,
            "engine": engine,
            "trace_path": trace_path,
        }

    def draft(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Backward-compatible alias for translate()."""
        return self.translate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.translate(*args, **kwargs)


def _generate_draft_sql(question: str, engine: str, model: str, db_info: Optional[str] = None) -> str:
    try:
        import litellm  # type: ignore
    except Exception as e:
        raise RuntimeError("litellm is required for tkboost.sql() draft generation.") from e

    payload = f"[QUESTION]\n{question.strip()}\n\n[ENGINE]\n{engine}\n"
    if db_info:
        payload += "\n[DB_INFO]\n" + db_info.strip() + "\n"
    prompt = (
        "Generate a single executable SQL query for the question. "
        "Return SQL only (prefer ```sql fenced block```), no explanation."
    )
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert SQL assistant."},
            {"role": "user", "content": prompt + "\n\n" + payload},
        ],
    )
    try:
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        content = getattr(resp.choices[0].message, "content", "")
    sql_text = _extract_sql_from_text(content or "")
    if not sql_text:
        raise RuntimeError("Unable to generate draft SQL from model.")
    return sql_text


def sql(
    question: Optional[str] = None,
    draft: Optional[str] = None,
    executor: Optional[Executor] = None,
    store: Optional[Union[str, TKStore]] = None,
    model: Optional[str] = None,
    db_name: Optional[str] = None,
    db_info: Optional[str] = None,
    use_llm_filtering: bool = False,
    max_turns: int = 10,
    min_probes: int = 3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Refine SQL using tribal knowledge from a store.

    Uses the CTE refiner's multi-turn agentic loop for iterative
    refinement with DB probing.  Falls back to single-shot refinement
    when no SQLite executor is available.

    If `draft` is not provided, a draft is generated from `question`
    using init() credentials.
    """
    store_path = store.path if isinstance(store, TKStore) else store
    if not store_path:
        raise ValueError("store is required (path to tkstore CSV)")
    try:
        import litellm  # type: ignore
    except Exception as e:
        raise RuntimeError("litellm is required for tkboost.sql(). Install dependencies first.") from e

    effective_model = model or _STATE.get("model") or "azure/gpt-5"
    engine = _infer_engine_from_executor(executor)

    draft_sql = draft
    if not draft_sql:
        if not question:
            raise ValueError("Provide either draft or question")
        draft_sql = _generate_draft_sql(question=question, engine=engine, model=effective_model, db_info=db_info)

    retriever = MemoryRetriever(store_path)

    # --- Multi-turn per-CTE refinement via cte_refiner (SQLite only) ---
    db_path = getattr(executor, "db_path", None) if executor else None
    if db_path and isinstance(executor, SQLiteExecutor):
        from src.agents.cte_refiner import run_refiner
        from src.utils.agent_utils import (
            parse_ctes_from_sql, rebuild_sql_from_ctes, extract_goal_from_cte_body,
        )

        user_q = question or "Refine the SQL query to return correct results."
        ctes, remainder_sql = parse_ctes_from_sql(draft_sql)
        all_verdicts: List[Dict[str, Any]] = []
        all_rules: List[Dict[str, Any]] = []
        total_rule_count = 0

        def _rules_block_for(sql_text: str) -> tuple:
            """Retrieve rules for a SQL fragment and build the text block."""
            frag_rules = retriever.retrieve(
                sql_text=sql_text,
                generic_only=False,
                use_llm_filtering=use_llm_filtering,
                llm_model=effective_model,
                db=db_name,
            )
            lines = []
            for r in frag_rules[:40]:
                txt = (r.get("rule") or "").strip()
                if txt:
                    lines.append(f"- {txt}")
            block = "\n".join(lines) if lines else "- No matching tribal knowledge rules found."
            return frag_rules, block

        if ctes:
            turns_per_cte = max(max_turns // max(len(ctes) + 1, 1), 5)

            for idx_cte, c in enumerate(ctes):
                cte_name = c.get("name") or ""
                cte_body = c.get("body") or ""
                cte_sql = f"WITH {cte_name} AS (\n{cte_body}\n)"
                goal_comment = extract_goal_from_cte_body(cte_body, cte_name)

                cte_rules, cte_rules_block = _rules_block_for(cte_sql)
                all_rules.extend(cte_rules)
                total_rule_count += len(cte_rules)

                cte_goal = (
                    f"{goal_comment}\n\n"
                    "Use these tribal knowledge rules as guidance:\n\n"
                    f"{cte_rules_block}"
                )

                prev_blocks = []
                for pc in ctes[:idx_cte]:
                    pname = pc.get("name") or ""
                    pbody = pc.get("body") or ""
                    pgoal = extract_goal_from_cte_body(pbody, pname)
                    prev_blocks.append(f"-- CTE: {pname}\n-- Goal: {pgoal}\nWITH {pname} AS (\n{pbody}\n)")
                previous_ctes_text = "\n\n".join(prev_blocks).strip() or None

                if verbose:
                    print(f"\n--- Refining CTE {idx_cte+1}/{len(ctes)}: {cte_name} "
                          f"({len(cte_rules)} rules retrieved) ---")

                verdict = run_refiner(
                    instance_id="",
                    db_id="",
                    user_query=user_q,
                    cte_text=cte_sql,
                    cte_goal=cte_goal,
                    previous_ctes=previous_ctes_text,
                    model=effective_model,
                    max_turns=turns_per_cte,
                    min_required_sql=min_probes,
                    verbose=verbose,
                    db_path=db_path,
                )
                all_verdicts.append({"cte": cte_name, "verdict": verdict})

                if verdict.get("status") == "issues" and verdict.get("suggested_fix_sql"):
                    fix_sql = verdict["suggested_fix_sql"]
                    fixed_ctes, _ = parse_ctes_from_sql(fix_sql)
                    if fixed_ctes:
                        c["body"] = fixed_ctes[0].get("body") or cte_body
                    else:
                        c["body"] = fix_sql

            # Final SELECT refinement
            if remainder_sql and remainder_sql.strip():
                full_sql = rebuild_sql_from_ctes(ctes, remainder_sql)
                rem_rules, rem_rules_block = _rules_block_for(full_sql)
                all_rules.extend(rem_rules)
                total_rule_count += len(rem_rules)

                rem_goal = (
                    f"Final SELECT using {len(ctes)} CTE(s)\n\n"
                    "Use these tribal knowledge rules as guidance:\n\n"
                    f"{rem_rules_block}"
                )
                prev_blocks = []
                for c in ctes:
                    cn = c.get("name") or ""
                    cb = c.get("body") or ""
                    cg = extract_goal_from_cte_body(cb, cn)
                    prev_blocks.append(f"-- CTE: {cn}\n-- Goal: {cg}\nWITH {cn} AS (\n{cb}\n)")
                previous_ctes_text = "\n\n".join(prev_blocks).strip() or None

                if verbose:
                    print(f"\n--- Refining final SELECT ({len(rem_rules)} rules retrieved) ---")

                final_verdict = run_refiner(
                    instance_id="",
                    db_id="",
                    user_query=user_q,
                    cte_text=full_sql,
                    cte_goal=rem_goal,
                    previous_ctes=previous_ctes_text,
                    model=effective_model,
                    max_turns=turns_per_cte,
                    min_required_sql=min_probes,
                    verbose=verbose,
                    db_path=db_path,
                )
                all_verdicts.append({"cte": "_final_select", "verdict": final_verdict})

                if final_verdict.get("status") == "issues" and final_verdict.get("suggested_fix_sql"):
                    refined_sql = final_verdict["suggested_fix_sql"]
                else:
                    refined_sql = rebuild_sql_from_ctes(ctes, remainder_sql)
            else:
                refined_sql = rebuild_sql_from_ctes(ctes, remainder_sql)

        else:
            # No CTEs: single-pass refinement on the whole SQL
            all_rules_list, rules_block = _rules_block_for(draft_sql)
            all_rules.extend(all_rules_list)
            total_rule_count = len(all_rules_list)

            cte_goal = (
                "Refine this SQL query so it returns the correct result. "
                "Use these tribal knowledge rules as guidance:\n\n"
                f"{rules_block}"
            )
            verdict = run_refiner(
                instance_id="",
                db_id="",
                user_query=user_q,
                cte_text=draft_sql,
                cte_goal=cte_goal,
                model=effective_model,
                max_turns=max_turns,
                min_required_sql=min_probes,
                verbose=verbose,
                db_path=db_path,
            )
            all_verdicts.append({"cte": "_full_query", "verdict": verdict})

            refined_sql = draft_sql
            if verdict.get("status") == "issues" and verdict.get("suggested_fix_sql"):
                refined_sql = verdict["suggested_fix_sql"]

        execution: Dict[str, Any] = {"ok": None, "error": None, "preview_headers": None, "preview_rows": None}
        if executor is not None:
            try:
                headers, rows = executor.execute(refined_sql)
                execution["ok"] = True
                execution["preview_headers"] = headers
                execution["preview_rows"] = [list(r) for r in rows[:10]]
            except Exception as e:
                execution["ok"] = False
                execution["error"] = str(e)

        seen_ids: set = set()
        unique_rules: List[Dict[str, Any]] = []
        for r in all_rules:
            mid = r.get("mem_id")
            if mid not in seen_ids:
                seen_ids.add(mid)
                unique_rules.append(r)

        return {
            "draft_sql": draft_sql,
            "refined_sql": refined_sql,
            "rule_count": total_rule_count,
            "rules_used": unique_rules[:40],
            "execution": execution,
            "verdicts": all_verdicts,
        }

    # --- Fallback: single-shot refinement (non-SQLite or no executor) ---
    fb_rules = retriever.retrieve(
        sql_text=draft_sql,
        generic_only=False,
        use_llm_filtering=use_llm_filtering,
        llm_model=effective_model,
        db=db_name,
    )
    fb_lines = []
    for r in fb_rules[:40]:
        txt = (r.get("rule") or "").strip()
        if txt:
            fb_lines.append(f"- {txt}")
    fb_rules_block = "\n".join(fb_lines) if fb_lines else "- No matching tribal knowledge rules found."

    refine_prompt = (
        "You are a SQL refiner. Improve the DRAFT SQL using only relevant tribal knowledge rules.\n"
        "Return ONLY the corrected SQL (prefer ```sql fenced block```), no explanation.\n\n"
        f"[DRAFT_SQL]\n{draft_sql}\n\n"
        f"[TRIBAL_KNOWLEDGE_RULES]\n{fb_rules_block}\n"
    )
    resp = litellm.completion(
        model=effective_model,
        messages=[
            {"role": "system", "content": "You produce executable SQL only."},
            {"role": "user", "content": refine_prompt},
        ],
    )
    try:
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        content = getattr(resp.choices[0].message, "content", "")
    refined_sql = _extract_sql_from_text(content or "") or draft_sql

    execution = {"ok": None, "error": None, "preview_headers": None, "preview_rows": None}
    if executor is not None:
        try:
            headers, rows = executor.execute(refined_sql)
            execution["ok"] = True
            execution["preview_headers"] = headers
            execution["preview_rows"] = [list(r) for r in rows[:10]]
        except Exception as e:
            execution["ok"] = False
            execution["error"] = str(e)

    return {
        "draft_sql": draft_sql,
        "refined_sql": refined_sql,
        "rule_count": len(fb_rules),
        "rules_used": fb_rules[:40],
        "execution": execution,
    }


__all__ = [
    "TKStoreEntry",
    "TKStore",
    "SQLAgent",
    "init",
    "generate",
    "sql",
    "Executor",
    "SQLiteExecutor",
    "BigQueryExecutor",
    "SnowflakeExecutor",
    "PostgresExecutor",
]

