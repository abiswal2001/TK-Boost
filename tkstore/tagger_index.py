import os
import re
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import litellm
from . import config
from .provider_env import log_provider


def _try_parse(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        first = s.find('{')
        last = s.rfind('}')
        if first != -1 and last != -1 and last > first:
            candidate = s[first:last+1]
            return json.loads(candidate)
    except Exception:
        pass
    return None


def generate_tagged_memories_json(
    instance_id: str,
    user_query: str,
    db_name: Optional[str],
    gold_sql: str,
    agent_sql: str,
    clean_summary: str,
    database_memories: List[str],
    generic_memories: List[str],
    evidence: Optional[str] = None,
    minimal_required_edits: Optional[List[str]] = None,
    model: str = "azure/o4-mini",
    verbose: bool = False,
    multiturn: bool = True,
    max_retries: int = 1,
):
    """Call LLM to produce a tagged JSON object for the provided memories.

    The function retries once if the LLM returns non-JSON text. It prints
    the resulting JSON (pretty) when verbose=True and returns the parsed dict.
    """
    # Final prompt: instruct LLM to emit indexed memories in strict CSV-ready JSON format
    prompt = (
        "Purpose:\n"
        "You are an expert SQL memory tagger. Your goal is to take the provided CLEAN_SUMMARY,\n"
        "DATABASE_MEMORIES (database-specific validation rules), and GENERIC_MEMORIES (general SQL principles) (these are input texts) and produce a single VALID JSON\n"
        "object that preserves the input memories verbatim and adds explicit, grounded tags plus an\n"
        "`index_rows` array ready for CSV ingestion. DO NOT rewrite or shorten the input memories — only tag them.\n\n"
        "RATIONALE (read carefully):\n"
        "The `search_keywords` and other tags you produce will be used by a downstream refiner/searcher to\n"
        "rapidly decide whether a stored memory/rule applies to a new CTE or SQL snippet. Matching will be\n"
        "performed primarily by exact token matching and simple regex over SQL text (for SQL operations and\n"
        "data-characteristics), or by table/column name matching (for table/column tags). Therefore tokens must be standard SQL-style\n"
        "phrases (e.g., 'group by', 'left join', 'strftime', 'avg') and should be grounded in the provided inputs\n"
        "(AGENT_SQL or GOLD_SQL or the memory text). Do NOT invent non-standard tokens. The downstream system\n"
        "will first filter by these tokens and then optionally run regex probes for confirmation.\n\n"
        "NOTE: The reserved token 'NA' is used only for storing a question-specific CLEAN_SUMMARY row that MUST NOT be used for search/matching. Taggers should never emit 'NA' for normal index rows.\n\n"
        "MANDATES (follow exactly):\n"
        " 1) Return VALID JSON ONLY. No surrounding text, no commentary.\n"
        " 2) Preserve `database_memories` and `generic_memories` exactly as provided (do not paraphrase or shorten).\n"
        " 3) Top-level keys must include: `instance_id`, `user_query`, `db`, `clean_summary`,\n"
        "    `database_memories`, `generic_memories`, and `index_rows`.\n"
        " 4) `index_rows` MUST be an array of objects with EXACT fields (order not important):\n"
        "     - `scope`: 'db' (for DATABASE_MEMORIES) or 'generic' (for GENERIC_MEMORIES). Note: 'question' scope is reserved for CLEAN_SUMMARY only (non-searchable).\n"
        "     - `sql_operations`: array of short lowercase operation names (examples: 'join','aggregation','window','filter')\n"
        "     - `table`: exact table name as appears in inputs, or the literal 'all' when the rule applies generally, or 'all' if not grounded\n"
        "     - `column`: exact column name as 'table.column' if grounded, or 'all', or 'all'\n"
        "     - `data_type`: one of ['str','int','numeric','date','bool','all','unspecified']\n"
        "     - `nulls`: one of ['Yes','No','all']\n"
        "     - `applies_when`: short condition string or 'all' (if always applicable)\n"
        "     - `rule`: the original memory text (preserve DESCRIPTION and EXAMPLE parts).\n\n"
        "GROUNDING RULES (CRITICAL):\n"
        " - Prefer deriving `table`/`column` from literals that appear verbatim in AGENT_SQL or GOLD_SQL. If none are found there, fall back to CLEAN_SUMMARY, DATABASE_MEMORIES (database_memories), then GENERIC_MEMORIES.\n"
        " - When AGENT_SQL or GOLD_SQL contains explicit table or table.column literals, populate `table` with the exact table name and `column` with the exact column (use 'table.column' when available). Prefer AGENT_SQL over GOLD_SQL when both are present.\n"
        " - If a generic memory text contains a literal table or column, move that literal into the corresponding `database_memories` entry (database-specific memory) and keep the generic memory text unchanged (generic memories should remain abstract).\n"
        " - If a value cannot be grounded, use 'all' for `table`/`column` and 'all' for `data_type`.\n\n"
        "CANONICAL VALUES & NORMALIZATION (must use exactly these forms):\n"
        " - `sql_operations` entries should be short, lowercase, and limited to verbs/nouns (e.g., 'join','aggregation','window','filter','cast').\n"
        " - `data_type` must be one of: 'str','int','numeric','date','bool','all','unspecified'.\n"
        " - `nulls` must be one of: 'Yes','No','all'.\n"
        " - `applies_when` should be a concise condition (like 'when join key nullable' or 'all').\n\n"
        "EXAMPLES (exact behavior; do not paraphrase in outputs):\n"
        "Inputs: generic_memories contains:\n"
        "  'Check that aggregate functions are applied to the correct base column, not to already summarized results. | DESCRIPTION: double-aggregation can inflate or distort true averages. | EXAMPLE: Prefer AVG(original_column) over AVG(aggregated_column).'\n"
        "Behavior: preserve that generic string as-is in `generic_memories`. Do NOT add table/column tags to it UNLESS the same literal table/column appears verbatim in CLEAN_SUMMARY, DATABASE_MEMORIES (database_memories), AGENT_SQL or GOLD_SQL.\n\n"
        "Inputs: database_memories (DATABASE_MEMORIES) contains a long rule with a specific table/column and example.\n"
        "Behavior: preserve the database memory text verbatim in `database_memories` and produce an `index_rows` entry with `table` and `column` grounded exactly to that literal.\n\n"
        "OUTPUT: Before returning the final JSON, perform TWO short reasoning steps enclosed in <think> tags to make your decisions explicit, then output ONLY the valid JSON object (no surrounding text):\n"
        "  1) <think>detected_ops: [..] — list canonical sql operations you detect across inputs (comma-separated).\n"
        "  2) <think>grounding: [..] — for each memory id, list a 1-3 token quoted evidence snippet (from inputs) you used to ground decisions.\n"
        "After the two <think> blocks, output ONLY the JSON object described above.\n"
        "\nSEARCH KEYWORDS (purpose & guidance):\n"
        " - Purpose: `search_keywords` are short, standard SQL-style tokens used by downstream tooling to decide whether a stored memory/rule applies to a new CTE or SQL snippet.\n"
        " - Requirement: For each index_row emit `search_keywords` as an array of tokens chosen from the canonical list below.\n"
        " - Grounding: Prefer tokens that appear verbatim in `AGENT_SQL` or `GOLD_SQL`; if none, you may use CLEAN_SUMMARY or the memory text. Do not invent tokens — if you cannot ground a canonical token, emit an empty `search_keywords` array.\n"
        "CANONICAL TOKENS (exact forms, lowercase, SQL-like):\n"
        " - joins: join, left join, inner join, cross join, outer join\n"
        " - join patterns: join multiplicity, coalesce, null_handling\n"
        " - aggregation: aggregation, avg, sum, count, count distinct, distinct, group by, having, double-aggregation\n"
        " - window/frame: window, rows between, range between, preceding, following, unbounded preceding, unbounded following, over\n"
        " - window functions / ranking: row_number, rank, dense_rank, ntile, percent_rank, cume_dist, lag, lead\n"
        " - rolling/moving: rolling, moving average, moving sum, moving count\n"
        " - filtering/ordering: where, filter, order by, limit, asc, desc\n"
        " - date/time & extraction: strftime, year, date, substr, date_parse, date_cast, timestamp\n"
        " - numeric / cast / precision: cast, round, numeric_precision, division, percent, ratio\n"
        " - string parsing: substr, substring, trim, like, upper, lower, concat\n"
        " - partitioning / over-partitioning: partition by, over, over-partitioning\n"
        " - counting / quantiles / percentiles: count, count distinct, ntile, quantile, percentile, median\n"
        " - subquery / correlated patterns: subquery, correlated_subquery, lateral, cte_definition\n"
        " - set / json / array ops: union, distinct, json_extract, unnest, pivot, unpivot\n"
        "\nDATA TYPE EXAMPLES (to pick exact tag values):\n"
        " - int: integer identifiers or counts (e.g., order_id, user_id) -> tag 'int'\n"
        " - numeric: monetary or measured values (e.g., payment_value, price, avg_rate) -> tag 'numeric'\n"
        " - str: textual categories or labels (e.g., category, name, code) -> tag 'str'\n"
        " - date: date/datetime/timestamp fields or extracted year/month (e.g., collision_date, created_at) -> tag 'date'\n"
        " - bool: binary flags (0/1 or TRUE/FALSE) (e.g., is_active) -> tag 'bool'\n"
        " - all: applies regardless of data type; 'unspecified' if type cannot be determined from inputs.\n"
        "\nNULLS GUIDANCE:\n"
        " - Use 'Yes' when the memory explicitly concerns NULLs/nullable behavior; use 'No' when it clearly applies to non-null fields; use 'all' otherwise.\n"
        "\nProduce the JSON now using only the inputs provided. Do NOT output anything else.\n"
    )

    inputs_block = {
        "instance_id": instance_id,
        "user_query": user_query,
        "db": db_name or "NOT_PRESENT",
        "gold_sql": gold_sql,
        "agent_sql": agent_sql,
        "evidence": evidence or "",
        "minimal_required_edits": minimal_required_edits or [],
        "clean_summary": clean_summary,
        "database_memories": database_memories,
        "generic_memories": generic_memories,
    }

    # normalize verbose into local boolean to avoid nested-scope warnings
    verbose = bool(verbose)

    messages = [
        {"role": "system", "content": "You are an expert SQL memory tagger."},
        {"role": "user", "content": prompt + "\nINPUTS:\n" + json.dumps(inputs_block, ensure_ascii=False, indent=2)},
    ]

    if verbose:
        print("\n" + "=" * 80)
        print(f"TAGGER START | instance_id={instance_id} | model={model} | multiturn={multiturn}")
        try:
            log_provider(model)
        except Exception:
            pass
        print("-" * 80)
        try:
            print("PROMPT (truncated):")
            print((prompt[:2000] + "...") if len(prompt) > 2000 else prompt)
            print("-" * 80)
            print("INPUTS (truncated):")
            print(json.dumps(inputs_block, ensure_ascii=False)[:2000])
        except Exception:
            print("(unable to print prompt/inputs)")
        print("=" * 80)

    # If multiturn requested perform two-step enrichment: produce JSON then augment with canonical tokens.
    if multiturn:
        # Turn 1: ask the tagger to produce the canonical JSON (with <think> tags as requested in prompt)
        resp_obj1 = litellm.completion(model=model, messages=messages)
        try:
            msg1 = resp_obj1["choices"][0]["message"]
        except Exception:
            try:
                msg1 = resp_obj1.choices[0].message
            except Exception:
                msg1 = None
        content1 = (msg1.get("content") if isinstance(msg1, dict) else getattr(msg1, "content", None)) if msg1 else None
        if verbose:
            print("[turn1] raw output (truncated):", (content1[:2000] + "...") if content1 and len(content1) > 2000 else content1)

        parsed1 = _try_parse(content1)
        if verbose:
            print("\n[TAGGER TURN1] raw content (first 4000 chars):")
            print((content1[:4000] + "...") if content1 and len(content1) > 4000 else content1)
            print("[TAGGER TURN1] parsed JSON?", bool(parsed1))

        if parsed1 is None and max_retries > 0:
            # retry once stricter
            if verbose:
                print("[tagger] first-turn parse failed; retrying with strict JSON instruction...")
            messages[1]["content"] = "STRICTLY OUTPUT VALID JSON ONLY. No surrounding text. " + messages[1]["content"]
            resp_obj1b = litellm.completion(model=model, messages=messages)
            try:
                msg1b = resp_obj1b["choices"][0]["message"]
            except Exception:
                try:
                    msg1b = resp_obj1b.choices[0].message
                except Exception:
                    msg1b = None
            content1b = (msg1b.get("content") if isinstance(msg1b, dict) else getattr(msg1b, "content", None)) if msg1b else None
            if verbose:
                print("[turn1 retry] raw output (truncated):", (content1b[:2000] + "...") if content1b and len(content1b) > 2000 else content1b)
            parsed1 = _try_parse(content1b)

        if parsed1 is None:
            # No valid JSON from turn1 — surface full content and abort
            if verbose:
                print("[TAGGER ERROR] TURN1 did not produce valid JSON. Raw content below:")
                print(content1)
            return None

        # Turn 2: augmentation pass — provide canonical tokens and ask tagger to enrich index_rows
        canonical_block = (
            "GROUNDING: Prefer tokens that appear verbatim in `AGENT_SQL` or `GOLD_SQL`; if none, you may use CLEAN_SUMMARY or the memory text. Do not invent tokens — if you cannot ground a canonical token, emit an empty `search_keywords` array.\n"
            "CANONICAL TOKENS (exact forms, lowercase, SQL-like):\n"
            " - joins: join, left join, inner join, cross join, outer join\n"
            " - join patterns: join multiplicity, coalesce, null_handling\n"
            " - aggregation: aggregation, avg, sum, count, count distinct, distinct, group by, having, double-aggregation\n"
            " - window/frame: window, rows between, range between, preceding, following, unbounded preceding, unbounded following, over\n"
            " - window functions / ranking: row_number, rank, dense_rank, ntile, percent_rank, cume_dist, lag, lead\n"
            " - rolling/moving: rolling, moving average, moving sum, moving count\n"
            " - filtering/ordering: where, filter, order by, limit, asc, desc\n"
            " - date/time & extraction: strftime, year, date, substr, date_parse, date_cast, timestamp\n"
            " - numeric / cast / precision: cast, round, numeric_precision, division, percent, ratio\n"
            " - string parsing: substr, substring, trim, like, upper, lower, concat\n"
            " - partitioning / over-partitioning: partition by, over, over-partitioning\n"
            " - counting / quantiles / percentiles: count, count distinct, ntile, quantile, percentile, median\n"
            " - subquery / correlated patterns: subquery, correlated_subquery, lateral, cte_definition\n"
            " - set / json / array ops: union, distinct, json_extract, unnest, pivot, unpivot\n"
        )

        enrich_prompt = (
        "TURN 2 - ENRICH: You will receive a JSON object (the tagger output) and must return ONLY a NEW valid JSON object where each `index_rows` entry has been enriched with canonical `sql_operations` (array) and `search_keywords` (array) selected from the canonical list below and strictly grounded in AGENT_SQL, GOLD_SQL, CLEAN_SUMMARY, database_memories (DATABASE_MEMORIES), and generic_memories.\n"
        "Purpose: Make each index_row fully searchable by refiners — prefer exhaustive, correct canonical tokens that a downstream searcher would match.\n"
        "Rules (strict):\n"
        " - Do NOT modify the original `rule` text or remove existing index_rows.\n"
        " - For each index_row: if `sql_operations` is missing or 'all', replace it with a best-effort list of canonical tokens grounded in the inputs; otherwise preserve and normalize it to canonical tokens from the provided canonical list.\n"
        " - Populate `search_keywords` as the subset of canonical tokens you grounded for that row (empty array if none).\n"
        " - For EVERY token you emit for an index_row, include a 1-3 token quoted evidence snippet used to ground that token in the separate <think>grounding: [...]</think> block (do NOT place evidence inside index_rows).\n"
        " - Prefer grounding evidence from AGENT_SQL or GOLD_SQL; only use CLEAN_SUMMARY or memory text when not present in SQL.\n"
        " - Do not invent tokens outside the canonical list; if unsure, leave token out rather than invent.\n\n"
            + canonical_block
            + "\nINPUT_JSON:\n" + json.dumps(parsed1, ensure_ascii=False) + "\n\nAGENT_SQL:\n" + (agent_sql or "") + "\n\nGOLD_SQL:\n" + (gold_sql or "") + "\n\nCLEAN_SUMMARY:\n" + (clean_summary or "") + "\n\nDATABASE_MEMORIES (database_memories):\n" + json.dumps(database_memories, ensure_ascii=False) + "\n\nGENERIC_MEMORIES:\n" + json.dumps(generic_memories, ensure_ascii=False) + "\n\nReturn ONLY the enriched JSON object."
        )

        resp_obj2 = litellm.completion(model=model, messages=[{"role": "system", "content": "You are an expert SQL memory tagger."}, {"role": "user", "content": enrich_prompt}])
        try:
            msg2 = resp_obj2["choices"][0]["message"]
        except Exception:
            try:
                msg2 = resp_obj2.choices[0].message
            except Exception:
                msg2 = None
        content2 = (msg2.get("content") if isinstance(msg2, dict) else getattr(msg2, "content", None)) if msg2 else None
        if verbose:
            print("\n[TAGGER TURN2] raw enrichment output (first 4000 chars):")
            print((content2[:4000] + "...") if content2 and len(content2) > 4000 else content2)

        parsed2 = _try_parse(content2)
        if parsed2 is None and max_retries > 0:
            if verbose:
                print("[tagger] enrichment parse failed; retrying with strict JSON instruction...")
            resp_obj2b = litellm.completion(model=model, messages=[{"role": "system", "content": "You are an expert SQL memory tagger."}, {"role": "user", "content": "STRICTLY OUTPUT VALID JSON ONLY. No surrounding text. " + enrich_prompt}])
            try:
                msg2b = resp_obj2b["choices"][0]["message"]
            except Exception:
                try:
                    msg2b = resp_obj2b.choices[0].message
                except Exception:
                    msg2b = None
            content2b = (msg2b.get("content") if isinstance(msg2b, dict) else getattr(msg2b, "content", None)) if msg2b else None
            if verbose:
                print("[TAGGER TURN2 retry] raw enrichment output (first 4000 chars):")
                print((content2b[:4000] + "...") if content2b and len(content2b) > 4000 else content2b)
            parsed2 = _try_parse(content2b)

        if parsed2 is None:
            # Enrichment failed — log and return the original parsed JSON so caller still has base tags.
            if verbose:
                print("[TAGGER WARNING] enrichment failed to produce valid JSON. Returning turn1 JSON. See raw enrichment output above for details.")
            return parsed1

        if verbose:
            print("[TAGGER TURN2] enrichment parsed successfully. Returning enriched JSON.")

        return parsed2

    # Fallback: single-call behavior (original logic)
    resp_obj = litellm.completion(model=model, messages=messages)
    # extract message object robustly
    try:
        msg_obj = resp_obj["choices"][0]["message"]
    except Exception:
        try:
            msg_obj = resp_obj.choices[0].message
        except Exception:
            msg_obj = None

    # pull content and reasoning_content if available
    resp = None
    resp_reasoning = None
    try:
        if isinstance(msg_obj, dict):
            resp = msg_obj.get("content") or None
            resp_reasoning = msg_obj.get("reasoning_content") or None
        else:
            resp = getattr(msg_obj, "content", None)
            resp_reasoning = getattr(msg_obj, "reasoning_content", None)
    except Exception:
        resp = None
        resp_reasoning = None

    if verbose:
        print("\n" + "=" * 80)
        print("TAGGER LLM RAW OUTPUT")
        print("-" * 80)
        try:
            print("CONTENT (truncated):")
            print((resp[:2000] + "...") if resp and len(resp) > 2000 else (resp or "<empty>"))
            if resp_reasoning:
                print("-" * 40)
                print("REASONING (truncated):")
                print((resp_reasoning[:2000] + "...") if len(resp_reasoning) > 2000 else resp_reasoning)
        except Exception:
            print("(unable to print LLM output)")
        print("=" * 80)

    parsed = _try_parse(resp)
    if parsed is None:
        # retry with stronger instruction
        if verbose:
            print("[tagger] first attempt not JSON, retrying with stricter instruction...")
        messages[1]["content"] = (
            "STRICTLY OUTPUT VALID JSON ONLY. No surrounding text. "
            + prompt
            + "\nINPUTS:\n"
            + json.dumps(inputs_block, ensure_ascii=False, indent=2)
        )
        # retry: call LLM again and capture message object
        resp_obj2 = litellm.completion(model=model, messages=messages)
        try:
            msg_obj2 = resp_obj2["choices"][0]["message"]
        except Exception:
            try:
                msg_obj2 = resp_obj2.choices[0].message
            except Exception:
                msg_obj2 = None
        try:
            if isinstance(msg_obj2, dict):
                resp2 = msg_obj2.get("content") or None
                resp2_reasoning = msg_obj2.get("reasoning_content") or None
            else:
                resp2 = getattr(msg_obj2, "content", None)
                resp2_reasoning = getattr(msg_obj2, "reasoning_content", None)
        except Exception:
            resp2 = None
            resp2_reasoning = None
        if verbose:
            try:
                print("[tagger] RAW LLM RESPONSE (2nd attempt, truncated):\n", (resp2[:2000] + "...") if resp2 and len(resp2) > 2000 else resp2)
                if resp2_reasoning:
                    print("[tagger] RAW LLM REASONING (2nd attempt, truncated):\n", (resp2_reasoning[:2000] + "...") if len(resp2_reasoning) > 2000 else resp2_reasoning)
            except Exception:
                pass
        parsed = _try_parse(resp2)

    if parsed is None:
        if verbose:
            print("[tagger] failed to parse JSON from LLM responses. Returning None.")
        return None

    if verbose:
        try:
            print("\n" + "=" * 80)
            print("TAGGER PARSED JSON (pretty):")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
            idx = parsed.get("index_rows") if isinstance(parsed, dict) else None
            if idx and isinstance(idx, list):
                print("-" * 80)
                print(f"index_rows count={len(idx)}; sample[0]=")
                print(json.dumps(idx[0], ensure_ascii=False))
        except Exception:
            print("(unable to pretty-print parsed JSON)")
        print("=" * 80)

    return parsed


def _infer_data_types_from_sql(sql_text: str) -> set:
    """Infer data types used in SQL based on patterns and functions.
    
    Returns a set of detected data types: {'str', 'int', 'numeric', 'date', 'bool', 'all'}
    Only detects types that are actively used in operations, not just present as join keys.
    """
    sql_l = (sql_text or "").lower()
    detected_types = set()
    
    # Date/timestamp patterns (active use: functions, comparisons, formatting)
    date_patterns = [
        r'\bdate\b.*(?:where|group|order|select|cast|convert)', r'\btimestamp\b', r'\bdatetime\b',
        r'strftime', r'to_date', r'to_timestamp', r'date_trunc',
        r'dateadd', r'datediff', r'\byear\b', r'\bmonth\b', r'\bday\b',
        r'_date\b.*(?:where|group|order|select|cast)', r'_at\b.*(?:where|group|order|select|cast)',
        r'collision_date', r'created_at', r'updated_at'
    ]
    if any(re.search(p, sql_l) for p in date_patterns):
        detected_types.add('date')
    
    # String patterns (active use: LIKE, functions, comparisons)
    string_patterns = [
        r'\bvarchar\b', r'\btext\b', r'\bstring\b',
        r'\blike\b', r'\bilike\b', r'\bconcat\b', r'\bsubstr\b', r'\bsubstring\b',
        r'\btrim\b', r'\blower\b', r'\bupper\b', r'\bregexp\b', r'\bcontains\b',
        r'\bcategory\b.*(?:where|group|order|select|cast)', r'\bcode\b.*(?:where|group|order|select|cast)',
        r'_name\b.*(?:where|group|order|select|cast)'
    ]
    if any(re.search(p, sql_l) for p in string_patterns):
        detected_types.add('str')
    
    # Numeric patterns (decimal/monetary) - most reliable detection
    numeric_patterns = [
        r'\bdecimal\b', r'\bfloat\b', r'\bnumeric\b', r'\breal\b', r'\bdouble\b',
        r'\bpayment_value\b', r'\bprice\b', r'\bamount\b', r'\bvalue\b(?!.*id)', r'\brate\b',
        r'\bavg\b', r'\bsum\b(?!.*\(.*count)', r'\bround\b', r'\bcast\b.*numeric', r'::numeric',
        r'::float', r'::decimal'
    ]
    if any(re.search(p, sql_l) for p in numeric_patterns):
        detected_types.add('numeric')
    
    # Integer patterns (only if actively used in operations, NOT just as join keys or GROUP BY)
    # Be very strict: only detect if integers are used in arithmetic, WHERE/HAVING comparisons, or COUNT(column)
    int_patterns = [
        # Explicit integer type declarations (in SELECT, WHERE, HAVING, CAST)
        r'(?:select|where|having|cast|::int)\s+.*\binteger\b',
        r'(?:select|where|having|cast|::int)\s+.*\bint\b(?!.*_id)',  # int but not _id
        r'\bbigint\b', r'\bsmallint\b',
        # COUNT(column) - COUNT on a specific column, not COUNT(*)
        r'\bcount\s*\(\s*[^*\s]',  # COUNT(column) not COUNT(*) or COUNT(DISTINCT)
        # Arithmetic operations on integer columns (must have operator before)
        r'(?:\+|\-|\*|\/)\s*(?:[^)]*\b_id\b|[^)]*\b(?:age|size|quantity|number|count)\b)',
        # WHERE/HAVING comparisons on integer columns (explicit comparisons)
        r'(?:where|having)\s+.*\b_id\b\s*(?:>|<|>=|<=|=|!=|between|in)',
        r'(?:where|having)\s+.*\b(?:age|size|quantity|number)\b\s*(?:>|<|>=|<=|=|!=|between|in)',
        # Integer in SELECT with operations (AVG, SUM, MAX, MIN on integer columns)
        r'(?:avg|sum|max|min)\s*\(\s*[^)]*\b_id\b',
        # Integer functions
        r'\bsize\s*\(',
    ]
    if any(re.search(p, sql_l) for p in int_patterns):
        detected_types.add('int')
    
    # Boolean patterns
    bool_patterns = [
        r'\bboolean\b', r'\bbool\b', r'\bis_active\b', r'\bis_null\b',
        r'\baggressive\b', r'_flag\b', r'= (?:true|false)\b', r'= 0\b', r'= 1\b'
    ]
    if any(re.search(p, sql_l) for p in bool_patterns):
        detected_types.add('bool')
    
    # Always include 'all' as fallback for broad rules
    detected_types.add('all')
    
    return detected_types


def _detect_null_handling_in_sql(sql_text: str) -> str:
    """Detect if SQL explicitly handles NULLs.
    
    Returns:
        'yes' if SQL has null handling (IS NULL, IS NOT NULL, COALESCE, etc.)
        'no' if SQL has no explicit null handling
    """
    sql_l = (sql_text or "").lower()
    
    null_handling_patterns = [
        r'\bis null\b', r'\bis not null\b', r'\bcoalesce\b', r'\bnvl\b',
        r'\bifnull\b', r'\bnullif\b', r'\bcase.*null', r'\bwhere.*null'
    ]
    
    if any(re.search(p, sql_l) for p in null_handling_patterns):
        return 'yes'
    
    return 'no'


def search_index_for_sql(
    sql_text: str, 
    index_csv_path: str, 
    generic_only: bool = True,
    db: Optional[str] = None,
    instance_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search memory index CSV for rules relevant to the provided SQL text.

    Args:
        sql_text: SQL query or CTE to validate
        index_csv_path: Path to the memory index CSV file
        generic_only: If True, only return generic-scoped rules
        db: Database name to filter rules by (optional)
        instance_id: Instance ID to filter rules by (optional)

    Matching strategy:
      - Match on sql_operations tokens (as before)
      - ALWAYS filter by data_type: if rule specifies a data_type (not 'all' or 'unspecified'),
        only match if SQL uses that data type
      - ALWAYS filter by nulls: if rule specifies nulls != 'all', only match if SQL null handling matches
      - Optionally filter by db and instance_id if provided

    Returns a list of dicts: {mem_id, instance_id, scope, sql_operations, table, column, data_type, nulls, rule, matches}
    where matches is a list of which checks triggered.
    """
    sql_l = (sql_text or "").lower()
    results: List[Dict[str, Any]] = []
    if not os.path.exists(index_csv_path):
        return results

    # Infer data types and null handling from SQL (once per query)
    sql_data_types = _infer_data_types_from_sql(sql_text)
    sql_null_handling = _detect_null_handling_in_sql(sql_text)

    with open(index_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mem_id = row.get('mem_id')
                instance_id = row.get('instance_id')
                scope = row.get('scope')
                ops_field = (row.get('sql_operations') or '')
                # ops stored as semicolon-separated
                ops = [o.strip().lower() for o in ops_field.split(';') if o.strip()]
                table = (row.get('table') or '').lower()
                column = (row.get('column') or '').lower()
                data_type = (row.get('data_type') or '').lower().strip()
                nulls = (row.get('nulls') or '').lower().strip()
                rule = row.get('rule') or ''

                matches: List[str] = []

                # Always skip 'question' scope (reserved for CLEAN_SUMMARY, non-searchable)
                if (scope or '').lower() == 'question':
                    continue
                # skip non-searchable rows (mems inserted as CLEAN_SUMMARY set sql_operations='NA')
                if ops_field.strip().upper() == 'NA':
                    continue
                
                # Scope and DB filtering logic:
                # Keep rows that are either:
                # 1. scope='generic' (always included), OR
                # 2. When db is provided: row's db matches the provided db (or row's db is 'all')
                scope_lower = (scope or '').lower()
                row_db = (row.get('db') or '').lower().strip()
                
                # If generic_only is True, only keep generic scope
                if generic_only:
                    if scope_lower != 'generic':
                        continue
                else:
                    # When generic_only is False and db is provided:
                    # Keep if scope='generic' OR db matches
                    if db is not None:
                        is_generic = (scope_lower == 'generic')
                        db_matches = (row_db == 'all' or row_db == db.lower())
                        
                        # Skip if NOT generic AND db doesn't match
                        if not is_generic and not db_matches:
                            continue
                
                # Optional instance_id filtering
                if instance_id is not None:
                    row_instance = (row.get('instance_id') or '').lower().strip()
                    # Match if the instance_id matches
                    if row_instance and row_instance != instance_id.lower():
                        continue

                # ============================================================
                # STRICT AND LOGIC: All three conditions must pass
                # ============================================================
                
                # 1. OPERATION MATCHING (OR logic within operations, but required overall)
                operation_matched = False
                matched_op_count = 0
                
                if not ops or 'all' in ops:
                    operation_matched = True
                    matches.append('op:all')
                else:
                    for op in ops:
                        if not op:
                            continue
                        # Check for word boundaries to avoid partial matches
                        if op in sql_l or re.search(r"\b" + re.escape(op) + r"\b", sql_l):
                            operation_matched = True
                            matched_op_count += 1
                            matches.append(f'op:{op}')
                
                # If no operation matched, skip this rule entirely (required condition)
                if not operation_matched:
                    continue
                
                # For generic scope rules with very common operations (select, where, join, from),
                # require at least 2 operation matches OR a specialized operation to reduce noise
                if scope_lower == 'generic' and matched_op_count == 1:
                    common_ops = {'select', 'where', 'join', 'from'}
                    specialized_ops = {
                        'to_timestamp', 'to_timestamp_ntz', 'date_trunc', 'to_date', 'to_char',
                        'st_distance', 'st_geogpoint', 'extract', 'unnest', '_table_suffix',
                        'safe_cast', 'lateral', 'flatten', 'pivot', 'window', 'row_number',
                        'partition by', 'isoweek', 'isoyear', 'case', 'coalesce'
                    }
                    matched_ops_set = {m.replace('op:', '') for m in matches if m.startswith('op:')}
                    
                    # If only matched a common op and no specialized op, skip this rule
                    if matched_ops_set.issubset(common_ops) and not matched_ops_set.intersection(specialized_ops):
                        continue
                
                # 2. DATA TYPE MATCHING (AND logic - must match if specified)
                data_type_matched = True
                if data_type and data_type not in ('all', 'unspecified', ''):
                    # Split semicolon-separated data types and check if ANY match
                    required_types = [dt.strip().lower() for dt in data_type.split(';') if dt.strip()]
                    matched_types = [dt for dt in required_types if dt in sql_data_types]
                    if not matched_types:
                        continue  # Skip: none of the required data types match (strict AND)
                    data_type_matched = True
                    for dt in matched_types:
                        matches.append(f'type:{dt}')
                else:
                    # data_type is 'all' or unspecified - always matches
                    data_type_matched = True
                
                # 3. NULL HANDLING MATCHING (AND logic - must match if specified)
                nulls_matched = True
                if nulls and nulls not in ('all', ''):
                    nulls_lower = nulls.lower().strip()
                    if nulls_lower == 'yes' and sql_null_handling != 'yes':
                        continue  # Skip: rule requires null handling but SQL doesn't have it
                    if nulls_lower == 'no' and sql_null_handling == 'yes':
                        continue  # Skip: rule is for non-null cases but SQL has null handling
                    # If we get here, null handling matches
                    nulls_matched = True
                    matches.append(f'nulls:{nulls_lower}')
                
                # All three conditions passed - include in results
                results.append({
                    'mem_id': mem_id,
                    'instance_id': instance_id,
                    'scope': scope,
                    'sql_operations': ops,
                    'table': table,
                    'column': column,
                    'data_type': data_type,
                    'nulls': nulls,
                    'rule': rule,
                    'matches': matches,
                })
            except Exception:
                continue
    return results


def _process_rule_chunk(cte_text: str, chunk_rules: List[Dict[str, Any]], rules_block: str, db_context: str, db: str, model: str) -> List[Dict[str, Any]]:
    """Process a single chunk of rules through LLM filtering.
    
    Helper function for _llm_filter_relevant_rules to handle chunked processing.
    """
    prompt = f"""You are helping filter validation rules for a SQL CTE validator.

BACKGROUND:
- These rules were created by analyzing training samples and identifying common mistakes and fixes
- Each rule tracks a specific type of error or tricky corner case that appeared in training data
- Rules guide the refiner to check specific aspects of CTEs (data types, null handling, joins, aggregations, etc.)
- The refiner needs relevant rules to properly refine a CTE - missing important rules significantly hurts refinement quality
- It's better to include a rule that might be relevant than to exclude one that could catch an error
{db_context}

GOAL:
Select rules that are likely to be helpful for validating the given CTE. When in doubt about a rule's relevance, INCLUDE it rather than exclude it. Only exclude rules that are clearly not applicable.

NOTE: The rules below are already PRE-ORDERED by relevance:
1. Database-specific rules for {db} come FIRST (highest priority, specialized operations first)
2. Generic rules come NEXT (specialized operations first, then general operations)

CTE TO VALIDATE:
```sql
{cte_text[:]}
```

CANDIDATE RULES ({len(chunk_rules)} total, pre-ordered by relevance):
{rules_block}

CRITERIA FOR SELECTION:
1. **Database-specific rules (scope='db')**: Strongly prefer db-specific rules for {db} unless clearly not applicable.

2. **CRITICAL: FUNCTION NAME CO-OCCURRENCE**: If a rule mentions any SQL function that appears in the CTE, include it. Function mentions in rule's recommended approach, anti-patterns, or examples all count.

3. **PRIORITIZE specialized terms** (TO_CHAR, DATE_TRUNC, TO_TIMESTAMP_NTZ, LATERAL, FLATTEN, ST_DISTANCE, etc.)

4. When in doubt, ERR ON THE SIDE OF INCLUSION

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
    "selected_indices": [1, 3, 5, ...],
    "reasoning": "Brief explanation"
}}

Where `selected_indices` is a list of rule numbers (1-based) from the candidate list.
Include rules where there's reasonable possibility (70%+) they could be relevant for validating this CTE."""

    try:
        messages = [
            {"role": "system", "content": "You are an expert SQL validation assistant that selects relevant validation rules."},
            {"role": "user", "content": prompt}
        ]
        resp = litellm.completion(model=model, messages=messages)
        content = resp["choices"][0]["message"]["content"].strip()
        
        # Parse JSON response
        import json
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group(0))
            selected_indices = result.get("selected_indices", [])
            
            # Convert to 0-based and filter rules from chunk
            selected_rules = []
            for idx in selected_indices:
                if 1 <= idx <= len(chunk_rules):
                    rule = chunk_rules[idx - 1].copy()
                    rule.pop('_original_idx', None)
                    selected_rules.append(rule)
            
            return selected_rules
        else:
            # Fallback: return all rules from chunk
            for rule in chunk_rules:
                rule.pop('_original_idx', None)
            return chunk_rules
    except Exception as e:
        print(f"[ERROR in chunk LLM filtering]: {e}")
        # On error, return all rules from chunk
        for rule in chunk_rules:
            rule.pop('_original_idx', None)
        return chunk_rules


def _llm_filter_relevant_rules(cte_text: str, candidate_rules: List[Dict[str, Any]], db: str = None, model: str = "azure/o4-mini") -> List[Dict[str, Any]]:
    """Use LLM to filter and select relevant rules for a given CTE.
    
    These rules were created by analyzing training samples and their fixes - they track
    common mistakes and tricky corner cases. The refiner needs relevant rules to
    properly check a CTE. This filter errs on the side of inclusion - when in doubt,
    rules are kept rather than excluded to avoid missing potential errors.
    
    Args:
        cte_text: The SQL CTE text to validate
        candidate_rules: List of candidate rules to filter
        db: Database name for the query (helps prioritize db-specific rules)
        model: LLM model to use for filtering
    
    Returns rules that could be helpful for validating this CTE.
    """
    if not candidate_rules or len(candidate_rules) == 0:
        return []
    
    # Separate rules into db-specific and generic for better ordering
    db_rules = []
    generic_rules = []
    
    for idx, rule in enumerate(candidate_rules):
        rule['_original_idx'] = idx  # Track original position
        if rule.get('scope') == 'db' and db and rule.get('db') == db:
            db_rules.append(rule)
        elif rule.get('scope') == 'db':
            # DB rule but different database - lower priority
            generic_rules.append(rule)
        else:
            generic_rules.append(rule)
    
    # Define operation categories by priority (niche/specialized first)
    specialized_ops = {
        'date_trunc', 'to_date', 'to_char', 'to_timestamp', 'extract',
        'st_distance', 'st_geogpoint', 'st_within', 'st_geogfromwkb',
        'unnest', 'lateral', 'flatten', 'pivot', 'unpivot',
        'window', 'row_number', 'rank', 'partition',
        'safe_cast', 'try_cast',
        '_table_suffix', 'wildcard',
        'cumulative', 'running',
        'regex', 'like',
    }
    
    def get_priority(rule):
        """Calculate priority score for a rule (higher = more specialized)."""
        ops_str = str(rule.get('sql_operations', '')).lower()
        # Check if any specialized operation is mentioned
        for spec_op in specialized_ops:
            if spec_op in ops_str:
                return 2  # High priority (specialized)
        return 1  # Normal priority (generic like join, select, where)
    
    # Sort each category by priority
    db_rules.sort(key=get_priority, reverse=True)
    generic_rules.sort(key=get_priority, reverse=True)
    
    # Combine: DB rules first (specialized first), then generic rules (specialized first)
    ordered_rules = db_rules + generic_rules
    
    # If we have too many rules, chunk them for better LLM processing
    # DB rules + high-priority generic rules should always be in first chunk
    CHUNK_SIZE = 15  # Process rules in chunks of 15 for better attention (avoid "lost in the middle" effect)
    
    if len(ordered_rules) > CHUNK_SIZE:
        # Process in chunks and combine results
        all_selected = []
        
        for chunk_start in range(0, len(ordered_rules), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(ordered_rules))
            chunk_rules = ordered_rules[chunk_start:chunk_end]
            
            # Build formatted rule list for this chunk
            rule_texts = []
            for idx, rule in enumerate(chunk_rules):
                mem_id = rule.get('mem_id', '?')
                scope = rule.get('scope', '?')
                db_name = rule.get('db', 'all')
                ops = rule.get('sql_operations', [])
                data_type = rule.get('data_type', 'all')
                nulls = rule.get('nulls', 'all')
                rule_text = rule.get('rule', '')[:500]
                
                rule_summary = (
                    f"[{idx+1}] mem_id={mem_id} | scope={scope} | db={db_name} | "
                    f"operations={', '.join(ops) if isinstance(ops, list) else ops} | "
                    f"data_type={data_type} | nulls={nulls}\n"
                    f"Rule: {rule_text}\n"
                )
                rule_texts.append(rule_summary)
            
            rules_block = "\n".join(rule_texts)
            db_context = f"\n\nDATABASE CONTEXT:\nThe CTE is querying database: {db}\nRules with scope='db' and db={db} are HIGHEST PRIORITY." if db else ""
            
            # Make LLM call for this chunk
            try:
                chunk_selected = _process_rule_chunk(cte_text, chunk_rules, rules_block, db_context, db, model)
                all_selected.extend(chunk_selected)
            except Exception as e:
                # On error for this chunk, include all rules from chunk
                print(f"[ERROR filtering chunk {chunk_start}-{chunk_end}]: {e}")
                all_selected.extend(chunk_rules)
        
        # Re-rank the selected rules: DB-specific rules first, then generic
        # This ensures the most relevant rules surface to the top
        def rerank_priority(rule):
            scope = rule.get('scope', '').lower()
            db_name = rule.get('db', '').lower()
            ops = rule.get('sql_operations', [])
            if isinstance(ops, str):
                ops = [o.strip().lower() for o in ops.split(';') if o.strip()]
            
            # Specialized operations get higher priority
            specialized_ops = {
                'to_timestamp', 'to_timestamp_ntz', 'to_timestamp_ltz', 'date_trunc', 'to_date', 'to_char',
                'st_distance', 'st_geogpoint', 'extract', 'unnest', '_table_suffix',
                'safe_cast', 'lateral', 'flatten', 'pivot', 'window', 'row_number',
                'partition by', 'isoweek', 'isoyear', 'segmentation', 'dicom'
            }
            has_specialized = any(op in specialized_ops for op in ops)
            
            # Priority scoring:
            # 1. DB-specific rules matching target DB + specialized ops: highest
            # 2. DB-specific rules matching target DB: high
            # 3. Generic rules with specialized ops: medium-high
            # 4. Generic rules: medium
            if scope == 'db' and db and db_name == db.lower():
                return 1000 + (100 if has_specialized else 0)
            elif scope == 'db':
                return 500 + (50 if has_specialized else 0)
            elif scope == 'generic' and has_specialized:
                return 100
            else:
                return 10
        
        # Sort by priority (highest first)
        all_selected.sort(key=rerank_priority, reverse=True)
        
        # Clean up tracking fields and return
        for rule in all_selected:
            rule.pop('_original_idx', None)
        return all_selected
    
    # If rules fit in one chunk, proceed with original logic
    # Build formatted rule list for LLM
    rule_texts = []
    for idx, rule in enumerate(ordered_rules):
        mem_id = rule.get('mem_id', '?')
        scope = rule.get('scope', '?')
        db_name = rule.get('db', 'all')
        ops = rule.get('sql_operations', [])
        data_type = rule.get('data_type', 'all')
        nulls = rule.get('nulls', 'all')
        rule_text = rule.get('rule', '')[:500]  # Truncate long rules
        
        rule_summary = (
            f"[{idx+1}] mem_id={mem_id} | scope={scope} | db={db_name} | "
            f"operations={', '.join(ops) if isinstance(ops, list) else ops} | "
            f"data_type={data_type} | nulls={nulls}\n"
            f"Rule: {rule_text}\n"
        )
        rule_texts.append(rule_summary)
    
    rules_block = "\n".join(rule_texts)
    
    db_context = f"\n\nDATABASE CONTEXT:\nThe CTE is querying database: {db}\nRules with scope='db' and db={db} are HIGHEST PRIORITY and should be strongly preferred." if db else ""
    
    prompt = f"""You are helping filter validation rules for a SQL CTE validator.

BACKGROUND:
- These rules were created by analyzing training samples and identifying common mistakes and fixes
- Each rule tracks a specific type of error or tricky corner case that appeared in training data
- Rules guide the refiner to check specific aspects of CTEs (data types, null handling, joins, aggregations, etc.)
- The refiner needs relevant rules to properly refine a CTE - missing important rules significantly hurts refinement quality
- It's better to include a rule that might be relevant than to exclude one that could catch an error
{db_context}

GOAL:
Select rules that are likely to be helpful for validating the given CTE. When in doubt about a rule's relevance, INCLUDE it rather than exclude it. Only exclude rules that are clearly not applicable.

NOTE: The rules below are already PRE-ORDERED by relevance:
1. Database-specific rules for {db} come FIRST (highest priority, specialized operations first)
2. Generic rules come NEXT (specialized operations first, then general operations)

CTE TO VALIDATE:
```sql
{cte_text[:]}
```

CANDIDATE RULES ({len(ordered_rules)} total, pre-ordered by relevance):
{rules_block}

CRITERIA FOR SELECTION:
1. **Database-specific rules (scope='db')**: If a rule has scope='db' and matches the database being queried, it should be strongly preferred unless it is clearly not applicable to the CTE. Database-specific rules capture patterns, errors, and best practices that are unique to that specific database.

2. **CRITICAL: FUNCTION NAME CO-OCCURRENCE**: If a rule mentions any SQL function (e.g., TO_CHAR, DATE_TRUNC, TO_TIMESTAMP_NTZ, LATERAL, FLATTEN) that appears in the CTE, the rule is likely relevant REGARDLESS of whether the function appears in the rule's:
   - Recommended approach ("use DATE_TRUNC for...")
   - Anti-pattern section ("instead of TO_CHAR...")
   - Examples or context
   EXAMPLE: CTE uses TO_CHAR(TO_TIMESTAMP_NTZ(...), 'YYYY-MM'). Rule says "use DATE_TRUNC instead of TO_CHAR for date grouping". This rule IS RELEVANT because it mentions both TO_CHAR (which is in the CTE) and TO_TIMESTAMP_NTZ (which is in the CTE), even though DATE_TRUNC is not in the CTE yet - the rule is identifying a correction opportunity.

3. Include rules that are directly relevant to operations/patterns in the CTE

4. Include rules whose data_type requirement matches what the CTE uses (or is 'all')

5. Include rules whose null handling requirement matches how the CTE handles nulls (or is 'all')

6. **PRIORITIZE rules that match on rare/specialized terms** (e.g., ISO, ISOWEEK, ISOYEAR, ST_DISTANCE, ST_GEOGPOINT, EXTRACT, UNNEST, _TABLE_SUFFIX, SAFE_CAST, cumulative, TO_TIMESTAMP, TO_TIMESTAMP_NTZ, DATE_TRUNC, TO_DATE, TO_CHAR, LATERAL, FLATTEN) - these specific terms are strong signals of relevance

7. **CRITICAL FOR ALTERNATIVE/CORRECTIVE RULES**: If a rule provides alternatives (e.g., "use X instead of Y", "prefer A over B"), check if EITHER pattern/function appears in the CTE

8. **CRITICAL FOR NEGATIVE RULES (AVOID/CAUTION)**: If a rule says to AVOID or CAUTION against a pattern, and that pattern IS PRESENT in the CTE, the rule is HIGHLY RELEVANT

9. When in doubt about relevance, ERR ON THE SIDE OF INCLUSION - it's better to have an extra rule than to miss catching an error

10. Only exclude rules that are clearly and obviously not applicable (e.g., spatial functions for non-spatial queries, array operations for scalar-only queries)

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
    "selected_indices": [1, 3, 5, ...],
    "reasoning": "Brief explanation of why these rules were selected"
}}

Where `selected_indices` is a list of rule numbers (1-based) from the candidate list.
IMPORTANT: Order the selected_indices by relevance with the following priorities:
1. Database-specific rules (scope='db') should come BEFORE generic rules (scope='generic')
2. Within each category (db/generic), order by relevance - most relevant rules first
Include rules where there's reasonable possibility (70%+) they could be relevant for validating this CTE."""
    
    try:
        messages = [
            {"role": "system", "content": "You are an expert SQL validation assistant that selects relevant validation rules."},
            {"role": "user", "content": prompt}
        ]
        resp = litellm.completion(model=model, messages=messages)
        content = resp["choices"][0]["message"]["content"].strip()
        
        # Parse JSON response
        import json
        # Try to extract JSON from response (might have markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group(0))
            selected_indices = result.get("selected_indices", [])
            
            # Convert to 0-based and filter rules from ordered list
            selected_rules = []
            for idx in selected_indices:
                if 1 <= idx <= len(ordered_rules):
                    rule = ordered_rules[idx - 1].copy()
                    # Remove temporary tracking field
                    rule.pop('_original_idx', None)
                    selected_rules.append(rule)
            
            return selected_rules
        else:
            # Fallback: if parsing fails, return all rules (better than losing them)
            # Clean up tracking fields
            for rule in ordered_rules:
                rule.pop('_original_idx', None)
            return ordered_rules
    except Exception as e:
        # On error, log and return all rules (better safe than sorry)
        print(f"[ERROR in LLM filtering]: {e}")
        import traceback
        traceback.print_exc()
        # Clean up tracking fields
        for rule in ordered_rules:
            rule.pop('_original_idx', None)
        return ordered_rules


class MemoryRetriever:
    """Thin retriever wrapper around the existing `search_index_for_sql`.

    Usage:
        mr = MemoryRetriever('/path/to/tkstore_bq.csv')
        rules = mr.retrieve(sql_text)
        rules = mr.retrieve(sql_text, use_llm_filtering=True)  # Optional LLM filtering
    """
    def __init__(self, index_csv_path: str):
        self.index_csv_path = index_csv_path

    def retrieve(
        self, 
        sql_text: str, 
        generic_only: bool = True, 
        use_llm_filtering: bool = False, 
        llm_model: str = "azure/gpt-4.1", 
        llm_only: bool = False,
        db: Optional[str] = None,
        instance_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant rules for SQL text.
        
        Args:
            sql_text: SQL query or CTE to validate
            generic_only: If True, only return generic-scoped rules
            use_llm_filtering: If True, use LLM to filter and select most relevant rules
            llm_model: Model to use for LLM filtering if enabled
            llm_only: If True, skip code-based filtering and let the LLM select from all rules
            db: Database name to filter rules by (optional)
            instance_id: Instance ID to filter rules by (optional)
        
        Returns:
            List of rule dictionaries
        """
        # If llm_only requested, bypass initial code-based filtering and load all index rows
        if llm_only:
            candidate_rules: List[Dict[str, Any]] = []
            if not os.path.exists(self.index_csv_path):
                return candidate_rules
            with open(self.index_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Scope and DB filtering logic:
                        # Keep rows that are either:
                        # 1. scope='generic' (always included), OR
                        # 2. When db is provided: row's db matches the provided db (or row's db is 'all')
                        scope = row.get('scope', '')
                        scope_lower = scope.lower() if scope else ''
                        row_db = (row.get('db') or '').lower().strip()
                        
                        # Skip 'question' scope
                        if scope_lower == 'question':
                            continue
                        
                        # If generic_only is True, only keep generic scope
                        if generic_only:
                            if scope_lower != 'generic':
                                continue
                        else:
                            # When generic_only is False and db is provided:
                            # Keep if scope='generic' OR db matches
                            if db is not None:
                                is_generic = (scope_lower == 'generic')
                                db_matches = (row_db == 'all' or row_db == db.lower())
                                
                                # Skip if NOT generic AND db doesn't match
                                if not is_generic and not db_matches:
                                    continue
                        
                        # Apply instance_id filtering if provided
                        if instance_id is not None:
                            row_instance = (row.get('instance_id') or '').lower().strip()
                            if row_instance and row_instance != instance_id.lower():
                                continue
                        
                        ops_field = (row.get('sql_operations') or '')
                        ops = [o.strip().lower() for o in ops_field.split(';') if o.strip()]
                        candidate_rules.append({
                            'mem_id': row.get('mem_id'),
                            'instance_id': row.get('instance_id'),
                            'scope': scope,
                            'sql_operations': ops,
                            'table': (row.get('table') or '').lower(),
                            'column': (row.get('column') or '').lower(),
                            'data_type': (row.get('data_type') or '').lower(),
                            'nulls': (row.get('nulls') or '').lower(),
                            'rule': row.get('rule') or '',
                        })
                    except Exception:
                        continue
            # If LLM filtering enabled, run the LLM selector on the full set
            if use_llm_filtering and candidate_rules:
                return _llm_filter_relevant_rules(sql_text, candidate_rules, db=db, model=llm_model)
            return candidate_rules

        # Default path: First, do initial filtering (operations, data_type, nulls)
        candidate_rules = search_index_for_sql(
            sql_text, 
            self.index_csv_path, 
            generic_only=generic_only,
            db=db,
            instance_id=instance_id
        )
        
        # If LLM filtering enabled and we have candidates, filter further
        if use_llm_filtering and candidate_rules:
            filtered_rules = _llm_filter_relevant_rules(sql_text, candidate_rules, db=db, model=llm_model)
            return filtered_rules
        
        return candidate_rules


class MemoryIndex:
    """Simple CSV-backed index manager for appending tagger output to disk.

    This mirrors the CSV layout used by the original harness.
    """
    def __init__(self, index_path: str):
        self.idx_path = Path(index_path)
        self.header = ["mem_id", "instance_id", "scope", "sql_operations", "table", "column", "data_type", "nulls", "rule"]

    def append_tagged(self, tagged_obj: Dict[str, Any], out_dir: str, instance_id: str, verbose: bool = True) -> None:
        # store global memory index in configured memory folder
        idx_path = Path(config.MEMORY_INDEX_PATH)
        header = self.header
        rows = []

        # If the LLM provided explicit index_rows, prefer them
        index_rows = tagged_obj.get("index_rows") if isinstance(tagged_obj, dict) else None
        if index_rows and isinstance(index_rows, list):
            for ir in index_rows:
                try:
                    scope = ir.get("scope", "db")  # Default to 'db' for database memories if not specified
                    ops = ir.get("sql_operations", [])
                    if not isinstance(ops, list):
                        ops = [ops]
                    ops_str = ";".join([str(o).lower() for o in ops]) if ops else "all"
                    table = ir.get("table", "all") or "all"
                    column = ir.get("column", "all") or "all"
                    data_type = ir.get("data_type", "unspecified") or "unspecified"
                    nulls = ir.get("nulls", "all") or "all"
                    rule = ir.get("rule", "")
                    rows.append([instance_id, scope, ops_str, table, column, data_type, nulls, rule.replace('\n', ' ')])
                except Exception:
                    rows.append([instance_id, "generic", "all", "all", "all", "unspecified", "all", json.dumps(ir, ensure_ascii=False)])
        else:
            # Fallback: best-effort extraction using available fields
            def _normalize_ops(ops: Any) -> str:
                if not ops:
                    return "all"
                if isinstance(ops, list):
                    return ";".join([str(o).lower() for o in ops])
                return str(ops).lower()

            def _extract_table_col(data_objs: List[str]) -> (str, str):
                table = "all"
                column = "all"
                for d in data_objs:
                    if not isinstance(d, str):
                        continue
                    d_low = d.lower()
                    if d_low.startswith("col:"):
                        rest = d_low[4:]
                        if "." in rest:
                            t, c = rest.split(".", 1)
                            return (t, c)
                        else:
                            return (rest, "all")
                    if d_low.startswith("table:"):
                        table = d_low[6:]
                return (table, column)

            def _infer_data_type(data_objs: List[str], text: str) -> str:
                ds = [d.lower() for d in data_objs]
                if any(x.startswith("data:time") or x == "data:time_series_column" for x in ds):
                    return "date"
                if any(x.startswith("data:text") or x == "data:textual_field" for x in ds):
                    return "str"
                if any(x.startswith("data:numeric") or x.startswith("data:aggregate") for x in ds):
                    return "numeric"
                return "NOT_PRESENT"

            def _infer_nulls(data_objs: List[str], text: str) -> str:
                ds = [d.lower() for d in data_objs]
                if any("nullable" in x or x == "data:nullable" for x in ds):
                    return "Yes"
                if "null" in text.lower():
                    return "Yes"
                return "No"

            for scope_key, scope_name in [("database_memories", "db"), ("generic_memories", "generic")]:
                mems = tagged_obj.get(scope_key) or []
                for mem in mems:
                    try:
                        if isinstance(mem, dict):
                            text = mem.get("text") or mem.get("rule") or json.dumps(mem, ensure_ascii=False)
                            tags = mem.get("tags") or {}
                        else:
                            text = str(mem)
                            tags = {}
                        sql_ops = tags.get("SQL_operations") or tags.get("sql_operations") or []
                        data_objs = tags.get("Data_objects") or tags.get("data_objects") or []
                        if not isinstance(sql_ops, list):
                            sql_ops = [sql_ops]
                        if not isinstance(data_objs, list):
                            data_objs = [data_objs]

                        ops = _normalize_ops(sql_ops)
                        table, column = _extract_table_col(data_objs)
                        data_type = _infer_data_type(data_objs, text)
                        nulls = _infer_nulls(data_objs, text)
                        rows.append([instance_id, scope_name, ops, table, column, data_type, nulls, text.replace('\n', ' ')])
                    except Exception:
                        rows.append([instance_id, scope_name, "all", "all", "all", "unspecified", "No", str(mem)])

        # write CSV (append)
        try:
            write_header = False
            start_idx = 0
            if not idx_path.exists():
                write_header = True
                start_idx = 0
            else:
                try:
                    first_line = None
                    with open(idx_path, 'r', encoding='utf-8') as rf:
                        for ln in rf:
                            if ln.strip():
                                first_line = ln.strip()
                                break
                    expected_header = ",".join(header)
                    if not first_line or (expected_header not in first_line and not first_line.startswith('mem_id')):
                        old = Path(idx_path).read_text(encoding='utf-8')
                        with open(idx_path, 'w', encoding='utf-8', newline='') as wf:
                            wf.write(expected_header + "\n")
                            wf.write(old)
                        existing = [ln for ln in old.splitlines() if ln.strip()]
                        start_idx = len(existing)
                    else:
                        try:
                            with open(idx_path, 'r', encoding='utf-8') as rf:
                                existing = [ln for ln in rf.read().splitlines() if ln.strip()]
                            start_idx = max(0, len(existing) - 1)
                        except Exception:
                            start_idx = 0
                except Exception:
                    start_idx = 0

            with open(idx_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                for i, r in enumerate(rows):
                    mem_id = start_idx + i
                    writer.writerow([mem_id] + r)
            if verbose:
                print(f"[index] appended {len(rows)} memory rows to {idx_path}")
        except Exception as e:
            if verbose:
                print(f"[index] failed to write index CSV: {e}")


