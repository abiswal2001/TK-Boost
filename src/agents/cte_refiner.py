#!/usr/bin/env python3
"""
CTE Refiner
==============

Refine a single CTE against the DB and its intended goal by letting the LLM freely alternate deep <think>/<sql> cycles before returning a <verdict_json>.
"""

import os
import re
import json
import argparse
import sqlite3
from pathlib import Path
import litellm


# --- Provider/Env ---
USE_OPENAI = True
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")

if USE_OPENAI:
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if OPENAI_API_BASE:
        os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
else:
    os.environ["AZURE_API_KEY"] = AZURE_API_KEY
    os.environ["AZURE_API_BASE"] = AZURE_API_BASE
    os.environ["AZURE_API_VERSION"] = AZURE_API_VERSION

AZURE_TO_OPENAI_MODEL = {
    "azure/gpt-4.1": "gpt-4.1",
    "azure/gpt-4o": "gpt-4o",
    "azure/o4-mini": "o4-mini",
}


def llm(model: str, messages: list, **kwargs):
    mapped = AZURE_TO_OPENAI_MODEL.get(model, model) if USE_OPENAI else model
    return litellm.completion(model=mapped, messages=messages, **kwargs)


def extract_tagged(content: str, tag: str):
    m = re.search(fr"<{tag}>(.*?)</{tag}>", content, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


class TraceLogger:
    """Simple trace logger for refiner execution"""
    
    def __init__(self, output_path: str = None):
        self.output_path = output_path
        self.sections = []
    
    def add_section(self, title: str, content: str):
        """Add a section to the trace"""
        if self.output_path:
            self.sections.append(f"=== {title} ===\n{content}\n")
    
    def save(self):
        """Save trace to file"""
        if self.output_path and self.sections:
            with open(self.output_path, 'w') as f:
                f.write('\n'.join(self.sections))


def get_database_path(instance_id: str, db_id: str) -> str:
    # Check mini_dev path first (if instance_id starts with "minidev")
    if instance_id.lower().startswith("minidev"):
        # Mini-dev databases are at: data/minidev/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite
        # Try relative path first (from project root)
        minidev_path = os.path.join("data", "minidev", "MINIDEV", "dev_databases", db_id, f"{db_id}.sqlite")
        if os.path.exists(minidev_path):
            return os.path.abspath(minidev_path)
        # If relative doesn't work, try absolute (should be same, but checking for safety)
        abs_path = os.path.abspath(minidev_path)
        if os.path.exists(abs_path):
            return abs_path
        raise FileNotFoundError(f"Mini-dev database not found: {minidev_path} (resolved: {abs_path})")
    
    # Original logic for other instances - use relative path
    base_folder = "data/spider2"
    example_folder = os.path.join(base_folder, instance_id)
    if os.path.isdir(example_folder):
        for file in os.listdir(example_folder):
            if file.endswith(".sqlite"):
                return os.path.join(example_folder, file)
        raise FileNotFoundError(f"No .sqlite file found in {example_folder}")
    fallback_path = f"{db_id}.sqlite"
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Database folder not found: {example_folder} and fallback {fallback_path} not found.")


def _extract_message_obj(resp):
    try:
        return resp["choices"][0]["message"]
    except Exception:
        try:
            return resp.choices[0].message
        except Exception:
            return None


def _message_to_dict(msg_obj):
    if msg_obj is None:
        return {"error": "no_message"}
    try:
        if hasattr(msg_obj, "model_dump"):
            return msg_obj.model_dump()
    except Exception:
        pass
    if isinstance(msg_obj, dict):
        return msg_obj
    data = {}
    for k in ("role", "content", "reasoning_content", "tool_calls", "function_call"):
        try:
            v = getattr(msg_obj, k, None)
        except Exception:
            v = None
        if v is not None:
            data[k] = v
    if not data:
        data["repr"] = repr(msg_obj)
    return data


def _extract_content(msg_obj, raw_dict):
    content = None
    reasoning = None
    if isinstance(raw_dict, dict):
        content = raw_dict.get("content")
        reasoning = raw_dict.get("reasoning_content")
    if not content:
        content = getattr(msg_obj, "content", None)
    if not reasoning:
        reasoning = getattr(msg_obj, "reasoning_content", None)
    text = (content or "").strip()
    if not text and reasoning:
        text = str(reasoning).strip()
    return text


def _format_table(headers, rows):
    if not rows:
        return "<empty>"
    headers = headers or [f"col{i+1}" for i in range(len(rows[0]))]
    sep = "-+-".join(["-" * len(h) for h in headers])
    lines = [" | ".join(headers), sep]
    for r in rows[:200]:
        lines.append(" | ".join([str(x) for x in r]))
    return "\n".join(lines)


def _parse_referenced_tables(cte_text: str):
    text = cte_text.lower()
    tables = set(re.findall(r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", text))
    return sorted(list(tables))


def _extract_first_cte_name(cte_text: str) -> str:
    try:
        m = re.search(r"with\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", cte_text or "", flags=re.IGNORECASE)
        if m:
            return m.group(1)
    except Exception:
        return ""
    return ""






def _contains_all(text: str, keywords: list) -> bool:
    try:
        low = (text or "").lower()
        return all(k in low for k in keywords)
    except Exception:
        return False


def _any_sql_matches(sql_texts: list, keyword_groups: list) -> bool:
    """Return True if any SQL in the list contains all keywords of at least one group.
    keyword_groups: list of lists; each inner list is an AND-group of keywords.
    """
    try:
        for sql in sql_texts or []:
            for group in keyword_groups:
                if _contains_all(sql, group):
                    return True
        return False
    except Exception:
        return False


SYSTEM_PROMPT_BASE  = """
You are an SQL CTE refiner agent.

Turn protocol (STRICT):
- Output exactly ONE block per turn: <think>...</think> OR <sql>...</sql> OR <verdict_json>...</verdict_json>.
- Never include more than one block in the same message.
- Never include <verdict_json> in the same message as <sql>.
- You may also emit executable SQL in a single fenced code block (```sql ... ```); treat that as the <sql> block for that turn.
 - FIRST TURN MUST BE a <sql>: SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name.
 - ONE SQL STATEMENT PER TURN. No semicolon-chained statements.

Goal:
- Given the user query, a CTE, and its intended goal, decide if the CTE is correct. Focus strictly on correctness (not performance).

Context:
- If [PREVIOUS_CTES] are provided, treat them as existing materialized views that the current CTE can reference. These are upstream CTEs that have already been validated and can be treated as available tables/views.
 - Be neutral and data-grounded: verify claims through SQL. Suggest fixes only when you have concrete, observed evidence (schemas, samples, DISTINCTs, NULL counts, join sanity). If uncertain, probe more; do not assume.
"""

SYSTEM_PROMPT_WITHOUT_PREDICTED = """
Behavior:
- Do deep exploration: alternate <think> (plan) and <sql> (one statement) for as many cycles as needed.
- Discover schema (tables/views), inspect relevant tables (PRAGMA, samples), and check DISTINCT/NULLs on referenced columns.
- Critically cross-check related tables (e.g., collisions, victims) if present: validate keys (case_id), join multiplicities, and whether the user query implies collision-level denominators vs party-level.
- If essential tables/joins/filters are missing, identify them and propose a minimal corrected CTE (or companion CTE) to align with the user query.
- Only when confident, emit a single <verdict_json> summarizing status, issues (if any), and a minimal suggested_fix.

Targeted checks when ranking/aggregation and year derivation may matter (adapt generically):
- Compare top-k by raw counts vs by normalized shares (e.g., percentage of annual total) per grouping unit; if results differ, flag a "ranking metric mismatch" and suggest the correct metric grounded in the user ask; include a probe showing the difference.
- Verify year derivation: STRFTIME('%Y', collision_date) vs any precomputed year fields; measure mismatch rate and flag if non-zero; prefer direct derivation from the date source when appropriate.
- When suggesting fixes, include 1–3 concise test SQLs that demonstrate the issue and confirm the correction.

Classification & denominator checks (generic):
- When labeling entities via JOIN+CASE, ensure denominator alignment:
  - Report base entity count vs classified entity count and explain any gap.
  - If using LEFT JOIN, verify whether NULLs from the joined side are excluded or mapped to an explicit 'unknown' bucket; do not silently default NULL to a negative class.
  - Prefer INNER JOIN or explicit WHERE <joined_col> IS NOT NULL for label assignment, unless an explicit 'unknown' category is produced.

Your <verdict_json> MUST follow this schema strictly:
{
  "status": "ok"|"issues",
  "issues": ["..."],
  "suggested_fix": "short rationale of the fix",
  "suggested_fix_sql": "FULL corrected CTE SQL text (WITH ... or SELECT ...)",
  "tests": [
    "SQL probe 1 that reproduces the issue",
    "SQL probe 2 that confirms the fix"
  ]
}
If ranking matters, suggested_fix_sql must compute a share = 1.0 * n / SUM(n) OVER (PARTITION BY <group>) and sort by share; if year matters, suggested_fix_sql must derive year via STRFTIME('%Y', <date_col>).
"""

SYSTEM_PROMPT_WITH_PREDICTED = """
Special Instructions (Predicted CTEs Available):
You have been given a reasoning model's plan for solving this query (see [PREDICTED_CTES_PLAN]).

IMPORTANT: Match CTEs based on their GOAL/PURPOSE, not their name. Just because a CTE name contains "parties" doesn't mean it should match a "party" CTE in the plan. Focus on what the CTE actually computes and its semantic purpose.

Validation steps:
1) Analyze the [CTE_GOAL] to understand its semantic purpose (e.g., "aggregate collision-level data", "flag helmet usage per party", "compute fatality rates").
2) Compare this purpose against the predicted CTE briefs in the plan. Match based on:
   - WHAT the CTE computes (not its name)
   - WHAT role it plays in solving the overall query
   - WHAT level of granularity it operates at (collision-level, party-level, etc.)
3) Do deep exploration using <think> and <sql> cycles:
   - Discover schema (tables/views), inspect relevant tables (PRAGMA, samples)
   - Check DISTINCT/NULLs on referenced columns
   - Critically cross-check related tables (e.g., collisions, victims) if present: validate keys (case_id), join multiplicities
   - Validate whether the user query implies collision-level denominators vs party-level
   - If ranking is involved (e.g., top-k), compare raw-count vs normalized-share variants; if different, explain and propose the right metric; verify year derivation consistency
4) After exploration, decide:
   - If the agent's CTE is correct (matches its stated goal and aligns with the PURPOSE of a predicted CTE) → status: "ok"
   - If the agent's CTE has issues OR doesn't align with any predicted CTE's PURPOSE → status: "issues"
5) When status is "issues", provide a suggested_fix that:
   - Fixes any data format/matching issues in the agent's CTE
   - OR rewrites the CTE to match the PURPOSE of the closest predicted CTE from the reasoning model's plan
   - OR proposes a companion CTE if essential tables/joins/filters are missing
   - Ensures the fix aligns with the overall predicted CTE architecture
   - Explains which predicted CTE's PURPOSE this should implement (not just name matching)

Only when confident, emit a single <verdict_json> containing:
{
  "status": "ok"|"issues",
  "issues": [...],  // List specific problems found
  "suggested_fix": "...",  // Short rationale
  "suggested_fix_sql": "FULL corrected CTE SQL (WITH ... or SELECT ...) that implements the fix and aligns with PURPOSE",
  "notes": "...",  // Explanation of alignment with predicted plan based on PURPOSE and data evidence
  "matched_predicted_cte": "..."  // Name of the predicted CTE whose PURPOSE this implements (if applicable)
}
// Include 1–3 simple test SQLs that reproduce the issue and confirm the correction (e.g., count vs percentage ranking diff; year mismatch rate)
"""


def run_refiner(instance_id: str, db_id: str, user_query: str, cte_text: str, cte_goal: str, predicted_ctes: str = None, previous_ctes: str = None, model: str = "azure/gpt-4.1", max_turns: int = 30, verbose: bool = True, trace_output_path: str = None, use_all_rules: bool = False, tribalknowledge_generic_only: bool = True, external_knowledge: str = None, schema_context: str = None):
    db_path = get_database_path(instance_id, db_id)
    
    # Initialize trace logger
    trace = TraceLogger(trace_output_path)
    
    if verbose:
        print("="*80)
        print(f"CTE REFINER | instance_id={instance_id} | db_path={db_path}")
        provider = "OpenAI" if USE_OPENAI else "Azure"
        print(f"Provider={provider} | Model={AZURE_TO_OPENAI_MODEL.get(model, model) if USE_OPENAI else model}")
        if predicted_ctes:
            print("Mode=WITH_PREDICTED_CTES")
        print("="*80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    verdict_data = None
    try:
        _ = _parse_referenced_tables(cte_text)

        # Build system prompt based on whether predicted CTEs are provided
        if predicted_ctes:
            system_prompt = SYSTEM_PROMPT_BASE + "\n" + SYSTEM_PROMPT_WITH_PREDICTED
        else:
            system_prompt = SYSTEM_PROMPT_BASE + "\n" + SYSTEM_PROMPT_WITHOUT_PREDICTED

        # Build user payload
        user_payload = "[USER_QUERY]\n" + (user_query or "").strip()
        
        # Log user query to trace
        trace.add_section("USER QUERY", user_query or "")
        
        # Add external knowledge and schema context from tkstore (like BigQuery/Snowflake)
        if external_knowledge:
            user_payload += "\n\n[EXTERNAL_KNOWLEDGE]\n" + external_knowledge.strip()
        
        if schema_context:
            user_payload += "\n\n[SCHEMA_CONTEXT]\n" + schema_context.strip()

        if previous_ctes:
            user_payload += "\n\n[PREVIOUS_CTES]\n" + previous_ctes.strip()

        if predicted_ctes:
            user_payload += "\n\n[PREDICTED_CTES_PLAN]\n" + predicted_ctes.strip()

        user_payload += "\n\n[CTE]\n" + cte_text.strip()
        user_payload += "\n\n[CTE_GOAL]\n" + cte_goal.strip()
        cte_name_hint = _extract_first_cte_name(cte_text)
        if cte_name_hint:
            user_payload += f"\n\n[CTE_NAME]\n{cte_name_hint}"
            user_payload += (
                "\n\n[Harness tip]\n"
                f"You may compile-test the CTE by concatenating [PREVIOUS_CTES] + this [CTE], then running: SELECT * FROM {cte_name_hint} LIMIT 10."
            )
        
        # Basic mandatory probes - semantic-specific rules now come from tkstore
        must_probes = [
            "PRAGMA table_info on all referenced base tables",
            "Sample rows from referenced tables (LIMIT 5)",
            "DISTINCT and NULL counts on key filter/derived columns",
        ]
        user_payload += "\n\n[MANDATORY_PROBES]\n- " + "\n- ".join(must_probes)

        if predicted_ctes:
            user_payload += "\n\nInstructions: Alternate <think> and <sql> (or a single fenced ```sql code block```) to explore thoroughly. One SQL per turn. Check alignment with predicted plan, validate collision-level vs party-level needs (case_id uniqueness), and related tables if present (e.g., collisions, victims). Only at the end, provide a single <verdict_json>."
        else:
            user_payload += "\n\nInstructions: Alternate <think> and <sql> (or a single fenced ```sql code block```) to explore thoroughly. One SQL per turn. Validate collision-level vs party-level needs (case_id uniqueness), and related tables if present (e.g., collisions, victims). Only at the end, provide a single <verdict_json>."

        if verbose:
            print("\n--- SYSTEM PROMPT ---")
            print(system_prompt)
            print("\n--- USER MESSAGE ---")
            print(user_payload)
            print("\n---------------------")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

        sql_executed = 0
        harness_executed = False
        no_sql_streak = 0
        min_required_sql = 8
        executed_sql_texts = []  # track probes actually run for hard gating
        for turn in range(1, max_turns + 1):
            if verbose:
                print(f"\n========== LLM TURN {turn} ==========")
            resp = llm(model, messages)
            msg_obj = _extract_message_obj(resp)
            raw = _message_to_dict(msg_obj)
            if verbose:
                print("[LLM RAW MESSAGE]:")
                try:
                    print(json.dumps(raw, indent=2))
                except Exception:
                    print(str(raw))
            content = _extract_content(msg_obj, raw)
            
            # Log LLM thinking to trace
            thinking = extract_tagged(content, "think")
            if thinking:
                trace.add_section(f"LLM THINKING (Turn {turn})", thinking)
            
            if not content:
                no_sql_streak += 1
                if no_sql_streak >= 2:
                    # Auto-fallback bootstrap: list tables/views
                    try:
                        if verbose:
                            print("\n[Auto SQL]: Listing tables/views")
                        auto_sql = "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
                        cursor.execute(auto_sql)
                        rows = cursor.fetchall()
                        headers = [d[0] for d in cursor.description] if cursor.description else None
                        table_text = _format_table(headers, rows)
                        if verbose:
                            print("\n[SQL RESULT]:\n" + table_text)
                        messages.append({"role": "user", "content": "SQL_RESULT_TABLE:\n" + table_text})
                        messages.append({"role": "user", "content": "Now emit a single <sql>: PRAGMA table_info(<likely_table>)"})
                        sql_executed += 1
                        executed_sql_texts.append(auto_sql)
                        continue
                    except Exception as e:
                        if verbose:
                            print("[SQL ERROR]:", str(e))
                messages.append({"role": "user", "content": "Please send exactly one <sql> now (start with listing tables or PRAGMA table_info(<table>))."})
                continue

            # If the assistant claims readiness, immediately ask for verdict
            lower = content.lower()
            if ("ready to deliver" in lower or "finalized assessment" in lower or "ready to" in lower) and "<verdict_json>" not in lower:
                messages.append({"role": "user", "content": "Now output <verdict_json> only (no other text)."})
                continue

            # Detect SQL/verdict blocks (both tag and fenced)
            sql_tag_blocks = re.findall(r"<sql>(.*?)</sql>", content, flags=re.DOTALL | re.IGNORECASE)
            sql_fence_blocks = re.findall(r"```\s*sql\s*([\s\S]*?)```", content, flags=re.DOTALL | re.IGNORECASE)
            sql_blocks = sql_tag_blocks + sql_fence_blocks
            # Generic fenced blocks as fallback (language-agnostic); only accept if looks like SQL
            if len(sql_blocks) == 0:
                generic_fences = re.findall(r"```\s*([\s\S]*?)```", content, flags=re.DOTALL)
                for blk in generic_fences:
                    blk_trim = blk.strip()
                    low = blk_trim.lower()
                    if any(k in low for k in ["select ", "pragma ", "with ", "explain "]):
                        sql_blocks.append(blk_trim)
            has_verdict = bool(re.search(r"<verdict_json>.*?</verdict_json>", content, flags=re.DOTALL | re.IGNORECASE))

            if len(sql_blocks) == 0 and not has_verdict:
                no_sql_streak += 1
            else:
                no_sql_streak = 0

            # Multiple SQL in one turn → execute first, nudge for next
            if len(sql_blocks) > 1:
                first_sql = sql_blocks[0].strip()
                if verbose:
                    print("\n[Executing SQL (first of multiple)]:\n" + first_sql)
                try:
                    cursor.execute(first_sql)
                    rows = cursor.fetchall()
                    headers = [d[0] for d in cursor.description] if cursor.description else None
                    table_text = _format_table(headers, rows)
                    if verbose:
                        print("\n[SQL RESULT]:\n" + table_text)
                    messages.append({"role": "user", "content": "SQL_RESULT_TABLE:\n" + table_text + "\nNote: One SQL per turn. Send the next probe in a new message."})
                    sql_executed += 1
                    executed_sql_texts.append(first_sql)
                    try:
                        if cte_name_hint and (f" from {cte_name_hint.lower()}" in first_sql.lower() or f" from [{cte_name_hint.lower()}]" in first_sql.lower()):
                            harness_executed = True
                    except Exception:
                        pass
                except Exception as e:
                    if verbose:
                        print("[SQL ERROR]:", str(e))
                    messages.append({"role": "user", "content": f"SQL_ERROR: {str(e)}\nNote: One SQL per turn. Send the next probe in a new message."})
                continue

            # Normal append
            messages.append({"role": "assistant", "content": content})

            # Single SQL block
            if len(sql_blocks) == 1:
                sql_text = sql_blocks[0].strip()
                if verbose:
                    print("\n[Executing SQL]:\n" + sql_text)
                try:
                    cursor.execute(sql_text)
                    rows = cursor.fetchall()
                    headers = [d[0] for d in cursor.description] if cursor.description else None
                    table_text = _format_table(headers, rows)
                    
                    # Log SQL execution to trace
                    trace.add_section(f"SQL QUERY (Turn {turn})", sql_text)
                    trace.add_section(f"SQL RESULT (Turn {turn})", table_text)
                    
                    if verbose:
                        print("\n[SQL RESULT]:\n" + table_text)
                    messages.append({"role": "user", "content": "SQL_RESULT_TABLE:\n" + table_text})
                    sql_executed += 1
                    executed_sql_texts.append(sql_text)
                    try:
                        if cte_name_hint and (f" from {cte_name_hint.lower()}" in sql_text.lower() or f" from [{cte_name_hint.lower()}]" in sql_text.lower()):
                            harness_executed = True
                    except Exception:
                        pass
                except Exception as e:
                    if verbose:
                        print("[SQL ERROR]:", str(e))
                    messages.append({"role": "user", "content": f"SQL_ERROR: {str(e)}"})
                continue

            # Verdict
            if has_verdict:
                # Basic minimum probes requirement
                if (
                    sql_executed < min_required_sql
                    or (cte_name_hint and not harness_executed)
                ):
                    needs = []
                    if sql_executed < min_required_sql:
                        needs.append(f"more probes ({sql_executed}/{min_required_sql})")
                    if cte_name_hint and not harness_executed:
                        needs.append("compile-test the CTE with SELECT * FROM CTE_NAME LIMIT 10")
                    messages.append({"role": "user", "content": "Before verdict, do: " + ", ".join(needs) + "."})
                    continue
                verdict_text = extract_tagged(content, "verdict_json")
                
                # Log verdict to trace
                if verdict_text:
                    trace.add_section(f"VERDICT JSON (Turn {turn})", verdict_text)
                
                try:
                    verdict_data = json.loads(verdict_text)
                    # Basic verdict validation - semantic-specific rules now come from tkstore
                    if verdict_data.get("status"):
                        need_rev = False
                        msg_needs = []
                        suggested_sql = verdict_data.get("suggested_fix_sql") or ""
                        tests_list = verdict_data.get("tests") or []
                        
                        # Require at least one test
                        if not tests_list:
                            need_rev = True
                            msg_needs.append("include 1–3 executable test SQLs in 'tests'")
                        
                        # LEFT JOIN + CASE must avoid silent default of NULLs
                        try:
                            if " left join " in suggested_sql.lower() and "case" in (suggested_sql or "").lower():
                                low = suggested_sql.lower()
                                has_unknown_bucket = "unknown" in low
                                has_not_null_filter = (" is not null" in low)
                                uses_inner_join = (" left join " not in low and " join " in low)
                                if not (has_unknown_bucket or has_not_null_filter or uses_inner_join):
                                    need_rev = True
                                    msg_needs.append("for LEFT JOIN + CASE labels: either use INNER JOIN, or WHERE <joined_col> IS NOT NULL, or create explicit 'unknown' bucket")
                        except Exception:
                            pass
                        
                        # Try to compile suggested_fix_sql as a single statement
                        if suggested_sql:
                            try:
                                cursor.execute(suggested_sql)
                                # consume minimal
                                _ = cursor.fetchone()
                            except Exception as e:
                                need_rev = True
                                msg_needs.append("make suggested_fix_sql a single executable SELECT/CTE statement: " + str(e))
                        
                        # Try executing tests (best-effort)
                        for test_sql in tests_list[:2]:
                            try:
                                cursor.execute(test_sql)
                                _ = cursor.fetchone()
                            except Exception as e:
                                need_rev = True
                                msg_needs.append("fix test SQL to be executable: " + str(e))
                        
                        if need_rev:
                            messages.append({"role": "user", "content": "Revise <verdict_json>: " + "; ".join(msg_needs) + "."})
                            verdict_data = None
                            continue
                    break
                except Exception:
                    messages.append({"role": "user", "content": "Please re-send <verdict_json> with valid JSON only (no extra text)."})
                    continue

            # Near end: force verdict request
            if turn >= max_turns - 1:
                messages.append({"role": "user", "content": "Time budget nearly exhausted. Output <verdict_json> only (no other text)."})
                continue

            # General nudge
            messages.append({"role": "user", "content": "Think, then send the next SQL probe (one per turn). Explore related tables (collisions, victims) and validate joins/keys against the user query."})
    
        # If we reached max turns without a verdict, force a final verdict
        if not verdict_data:
            if verbose:
                print(f"\n========== FORCED FINAL VERDICT (max turns reached) ==========")
            
            final_message = """TIME LIMIT REACHED. You MUST now provide a <verdict_json> based on ALL the exploration you have done so far.

CRITICAL: Even if you are not 100% certain, you MUST make a decision. Use all the SQL probes, schema exploration, and data analysis you have already performed to make the best possible assessment.

REQUIREMENTS:
- If you found ANY issues during your exploration (missing tables, incorrect joins, wrong aggregations, etc.), set status to "issues" and provide suggested_fix_sql
- If the CTE appears correct based on your exploration, set status to "ok" 
- You MUST provide suggested_fix_sql if status is "issues" - even if it's not perfect, provide your best attempt based on what you discovered
- Include 1-2 test SQLs in the 'tests' array that demonstrate the issue or confirm the fix

IMPORTANT: Your response must contain ONLY a valid JSON object wrapped in <verdict_json> tags. No other text. Example format:
<verdict_json>
{
  "status": "issues",
  "issues": ["missing table join"],
  "suggested_fix": "Add missing join to table X",
  "suggested_fix_sql": "SELECT ... FROM table1 JOIN table2 ...",
  "tests": ["SELECT COUNT(*) FROM table1", "SELECT COUNT(*) FROM table2"]
}
</verdict_json>

Do NOT ask for more exploration. Make a decision NOW based on what you already know."""
            
            messages.append({"role": "user", "content": final_message})
            
            try:
                resp = llm(model, messages)
                msg_obj = _extract_message_obj(resp)
                raw = _message_to_dict(msg_obj)
                content = _extract_content(msg_obj, raw)
                
                if verbose:
                    print("[FORCED VERDICT RESPONSE]:")
                    print(content)
                
                verdict_text = extract_tagged(content, "verdict_json")
                
                # Log forced verdict to trace
                trace.add_section("FORCED VERDICT RESPONSE", content)
                if verdict_text:
                    trace.add_section("FORCED VERDICT JSON", verdict_text)
                
                if verdict_text:
                    try:
                        verdict_data = json.loads(verdict_text)
                        if verbose:
                            print(f"[FORCED VERDICT SUCCESS]: {verdict_data.get('status', 'unknown')}")
                    except Exception as e:
                        if verbose:
                            print(f"[FORCED VERDICT JSON ERROR]: {e}")
                        # Save raw verdict text as a fallback - this might still be useful
                        if verbose:
                            print("[FORCED VERDICT]: Saving raw verdict text as fallback")
                        verdict_data = {
                            "status": "issues",
                            "issues": ["forced_verdict_json_parse_error"],
                            "suggested_fix": "Unable to parse verdict JSON - see raw_verdict_text for details",
                            "suggested_fix_sql": "",
                            "tests": [],
                            "raw_verdict_text": verdict_text,
                            "notes": f"JSON parse error: {str(e)}"
                        }
                else:
                    if verbose:
                        print("[FORCED VERDICT FAILED]: No verdict_json found in response")
                    # Save raw response as fallback
                    verdict_data = {
                        "status": "issues", 
                        "issues": ["forced_verdict_no_json_tags"],
                        "suggested_fix": "No verdict JSON provided - see raw_response for details",
                        "suggested_fix_sql": "",
                        "tests": [],
                        "raw_response": content,
                        "notes": "No <verdict_json> tags found in LLM response"
                    }
                    
            except Exception as e:
                if verbose:
                    print(f"[FORCED VERDICT ERROR]: {e}")
                verdict_data = None
    
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    # Save trace before returning
    trace.save()

    return verdict_data or {"status": "issues", "issues": ["no_verdict"], "suggested_fix": "", "notes": "max_turns reached"}


def main():
    p = argparse.ArgumentParser(description="Validate a CTE against its intended goal using LLM+DB probes")
    p.add_argument("--instance_id", required=True)
    p.add_argument("--db", required=False, default=None, help="DB id (optional; used for fallback name)")
    p.add_argument("--user-query", required=False, default=None, help="User main question inline")
    p.add_argument("--user-file", required=False, default=None, help="Path to file containing user question")
    p.add_argument("--cte", required=False, default=None, help="CTE text inline")
    p.add_argument("--cte-file", required=False, default=None, help="Path to file containing CTE text")
    p.add_argument("--goal", required=False, default=None, help="CTE goal/comment inline")
    p.add_argument("--goal-file", required=False, default=None, help="Path to file containing goal text")
    p.add_argument("--predicted-ctes", required=False, default=None, help="Predicted CTEs plan from reasoning model (inline)")
    p.add_argument("--predicted-ctes-file", required=False, default=None, help="Path to file containing predicted CTEs plan")
    p.add_argument("--previous-ctes", required=False, default=None, help="Previous CTEs context (name: description) to treat as materialized views (inline)")
    p.add_argument("--previous-ctes-file", required=False, default=None, help="Path to file containing previous CTEs context")
    p.add_argument("--model", default="azure/gpt-4.1")
    p.add_argument("--use-all-rules", action="store_true", help="Include all semantic-specific probes regardless of CTE content")
    p.add_argument("--verbose", action="store_true", help="Verbose terminal logging")
    p.add_argument("--out", default=None, help="Optional output JSON path")
    p.add_argument("--trace", default=None, help="Optional trace output path")
    args = p.parse_args()

    user_query = args.user_query
    if not user_query and args.user_file:
        user_query = Path(args.user_file).read_text(encoding="utf-8")

    cte_text = args.cte
    if not cte_text and args.cte_file:
        cte_text = Path(args.cte_file).read_text(encoding="utf-8")

    goal = args.goal
    if not goal and args.goal_file:
        goal = Path(args.goal_file).read_text(encoding="utf-8")

    predicted_ctes = args.predicted_ctes
    if not predicted_ctes and args.predicted_ctes_file:
        predicted_ctes = Path(args.predicted_ctes_file).read_text(encoding="utf-8")

    previous_ctes = args.previous_ctes
    if not previous_ctes and args.previous_ctes_file:
        previous_ctes = Path(args.previous_ctes_file).read_text(encoding="utf-8")

    if not cte_text or not goal or not user_query:
        print("❌ Provide user query, CTE text, and CTE goal (inline or via files)")
        return

    db_id = args.db or "unknown_db"

    verdict = run_refiner(args.instance_id, db_id, user_query, cte_text, goal, predicted_ctes=predicted_ctes, previous_ctes=previous_ctes, model=args.model, verbose=bool(args.verbose), trace_output_path=args.trace, use_all_rules=bool(args.use_all_rules))
    print(json.dumps(verdict, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
