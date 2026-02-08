import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import litellm

from .format_utils import format_csv_as_table, _is_csv_like
from .provider_env import log_provider
from .tagger_index import generate_tagged_memories_json
from . import config
import csv
from src.executors.factory import make_executor
from src.utils.agent_utils import infer_engine, load_external_knowledge


def load_snowflake_schema_context(db_name: str) -> Optional[str]:
    """Load compressed schema context for Snowflake database."""
    if not db_name:
        return None
    # Resolve path relative to project root (same as agent)
    project_root = Path(__file__).resolve().parent.parent
    schema_dir = project_root / "data" / "sf_schemas"
    schema_file = schema_dir / f"{db_name}.txt"
    if schema_file.exists():
        return schema_file.read_text(encoding='utf-8')
    return None


def load_bigquery_schema_context(db_name: str) -> Optional[str]:
    """Load compressed schema context for BigQuery database."""
    if not db_name:
        return None
    # Resolve path relative to project root (same as agent)
    project_root = Path(__file__).resolve().parent.parent
    schema_dir = project_root / "data" / "bq_schemas"
    schema_file = schema_dir / f"{db_name}.txt"
    if schema_file.exists():
        return schema_file.read_text(encoding='utf-8')
    return None


def _llm(model: str, messages: List[Dict[str, str]]) -> str:
    # Reuse litellm with existing OPENAI/AZURE env the project already sets
    # Map azure/* aliases to OpenAI equivalents when only OPENAI creds are present
    resp = litellm.completion(model=model, messages=messages)
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def generate_memory_diff_first_turn(
    instance_id: str,
    user_query: str,
    agent_cte_text: str,
    gold_sql_text: str,
    gold_result_csv_text: str,
    agent_full_sql_text: str,
    agent_result_csv_text: Optional[str] = None,
    processed_trace_text: Optional[str] = None,
    engine: Optional[str] = None,
    db_path_or_cred: Optional[str] = None,
    db_name: Optional[str] = None,
    external_knowledge: Optional[str] = None,
    max_turns: int = 6,
    model: str = "azure/o4-mini",
    verbose: bool = True,
    hint: Optional[str] = None,
) -> Optional[str]:
    """
    First-turn interaction: ask the LLM for a concise human-readable diff between
    the GOLD SQL and the AGENT SQL at the CTE level. Return plain text only.

    This function intentionally performs a single LLM call and returns the
    textual diff (or empty string) so we can iterate on the prompt/dialogue
    before building full rule objects.
    
    Args:
        engine: Database engine type ('sqlite', 'snowflake', 'bq', etc.). If None, inferred from instance_id.
        db_path_or_cred: For SQLite: database file path. For Snowflake/BigQuery: credential file path.
                        If None, will attempt to resolve from JSONL based on engine type.
        db_name: Database name (for Snowflake/BigQuery schema context loading). If None, will attempt to extract from JSONL.
    """
    # Infer engine from instance_id if not provided
    if engine is None:
        engine = infer_engine(instance_id)
    
    # Load schema context if needed (Snowflake or BigQuery)
    schema_context = None
    if engine == "snowflake":
        if db_name:
            schema_context = load_snowflake_schema_context(db_name)
            if schema_context and verbose:
                print(f"[Snowflake] Loaded schema context for db: {db_name}")
    elif engine == "bq" or engine == "bigquery":
        if db_name:
            schema_context = load_bigquery_schema_context(db_name)
            if schema_context and verbose:
                print(f"[BigQuery] Loaded schema context for db: {db_name}")
    
    # Build base prompt (common for all engines)
    base_prompt = (
        "You are an SQL analyst and MINIMAL-REPAIR agent. Operate interactively. For each proposed edit follow this strict protocol:\n"
        "  1) Emit a single <think>...</think> that MUST contain exactly one line starting with 'EDIT_PLAN:' describing the single minimal clause-level change you WILL apply (one short sentence).\n"
        "  2) After the EDIT_PLAN, emit exactly one <sql>...</sql> containing the minimal SQL edit to test. The SQL must be as small as possible (one clause change: WHERE, JOIN ON, GROUP BY, or conversion of AVG->SUM/COUNT(DISTINCT)).\n"
        "  3) For BigQuery: If your SQL uses CTEs, ensure NO struct prefixes (totals., hits., etc.) are used in CTEs that reference other CTEs - use simple column names only.\n"
        "The harness will execute your <sql> and return its result as SQL_RESULT. Inspect SQL_RESULT against the GOLD EXECUTION RESULT provided below.\n"
        "Note: If the SQL_RESULT contains only column headers and no data rows (an empty result set), you must carefully investigate and explain why the result is empty\n"
        "CRITICAL MATCHING REQUIREMENTS - READ CAREFULLY:\n"
        "- You can ONLY declare 'MATCH_OK' if SQL_RESULT matches GOLD EXACTLY - row by row, value by value\n"
        "- EXACT MATCH means: Same number of rows, same row order, same column values (within rounding tolerance of 0.01 for numeric values)\n"
        "- MANDATORY VALUE-BY-VALUE COMPARISON: You MUST explicitly compare EVERY numeric value and EVERY string value between SQL_RESULT and GOLD\n"
        "- For each row in SQL_RESULT, check the corresponding row in GOLD:\n"
        "  * Compare column 1: SQL_RESULT value vs GOLD value - are they the same? (within 0.01 for numbers)\n"
        "  * Compare column 2: SQL_RESULT value vs GOLD value - are they the same? (within 0.01 for numbers)\n"
        "  * Compare ALL columns - if ANY column value differs, it is NOT a match\n"
        "- Examples of NON-MATCHES that MUST be caught:\n"
        "  * max_duration: 1785.27 vs 2848.47 → NOT A MATCH (difference of 1063.2)\n"
        "  * total_trips: 34655569 vs 32861436 → NOT A MATCH (different counts)\n"
        "  * average_fare: 7.76 vs 5.54 → NOT A MATCH (different values)\n"
        "  * min_minutes: 0-4 vs 0-5 → NOT A MATCH (different bin boundaries)\n"
        "- If you see ANY differences in values (even if next_page matches), you MUST continue with more edits - DO NOT declare MATCH_OK\n"
        "- DO NOT declare MATCH_OK just because one column matches (e.g., next_page) - ALL columns must match\n"
        "- DO NOT declare MATCH_OK if numeric values differ by more than 0.01 - these are structural differences, not cosmetic\n"
        "- Only when SQL_RESULT and GOLD are identical (or within 0.01 tolerance for numeric values) should you declare MATCH_OK\n"
        "- If SQL_RESULT matches the GOLD exactly, respond with exactly two plain lines: 'MATCH_OK' and then 'MINIMAL_FIX: <one short sentence>' describing the minimal change you applied. Then stop.\n"
        "If not matched, repeat the protocol (one EDIT_PLAN then one SQL) until you either reach the SQL attempt budget or find a match.\n"
        "CRITICAL: If your SQL returns empty results but GOLD has data rows, DO NOT conclude that data doesn't exist. Instead, carefully examine the GOLD SQL to understand how it queries the data. The GOLD SQL proves the data exists - your query approach may be wrong (wrong table, wrong join, wrong filter condition, wrong date format, etc.). Always propose new SQL edits based on what you see in GOLD SQL.\n"
        "CRITICAL: If you cannot propose any further SQL edits, you MUST respond with 'NO_DIFF' (not just apologize or explain). Do NOT waste turns by repeating explanations without proposing SQL edits.\n"
        "\n"
        "*** MANDATORY: STRICT MINIMALITY AND INCREMENTAL REPAIR ***\n"
        "YOU MUST START FROM THE AGENT SQL AND MAKE MINIMAL INCREMENTAL EDITS:\n"
        "- Your <sql> MUST be based on the AGENT FULL SQL provided below, with only the specific clause(s) changed\n"
        "- DO NOT rewrite the entire query structure - only modify the specific clauses that are wrong\n"
        "- DO NOT copy the entire GOLD SQL structure - this defeats the purpose of minimal repair\n"
        "- Example: If the agent's WHERE clause is wrong, change ONLY that WHERE clause, keeping all CTEs and structure intact\n"
        "- Example: If the agent's JOIN condition is wrong, change ONLY the JOIN ON clause, keeping everything else\n"
        "- Example: If the agent's aggregation is wrong, change ONLY the aggregation function (e.g., AVG to SUM/COUNT), keeping the rest\n"
        "- If you find yourself writing a completely new query structure, STOP and instead identify the minimal change needed in the AGENT SQL\n"
        "- The goal is to show WHAT SPECIFIC CHANGE fixes the agent's SQL, not to replace it with GOLD\n"
        "- Only if you have tried multiple minimal edits and they all fail, then you may propose a more significant structural change (but explain why minimal edits didn't work)\n"
        "- FORBIDDEN: Copying the entire GOLD SQL and passing it as your edit - this is NOT minimal repair\n"
        "- FORBIDDEN: Rewriting all CTEs from scratch - start from agent's CTEs and fix only what's wrong\n"
        "FORBIDDEN_TAGS: Only use <think>...</think> and <sql>...</sql> for the interactive protocol. Do NOT emit other tags such as <solution>; such outputs will be treated as non-compliant.\n"
        "Important: In every <think> block include a single line starting with 'OBSERVED_DIFFS:' listing concrete differences you see between the latest SQL_RESULT and the GOLD result. Each observed difference MUST be tagged [STRUCTURAL] or [COSMETIC] (examples: 'OBSERVED_DIFFS: NULLS: interest_name in monthly_top_interest [STRUCTURAL]; ROW_COUNT:-2 [STRUCTURAL]; VALUE_SCALE: too many decimals [COSMETIC]; COLUMN_NAME: avg_balance_quadrillions vs average_balance_trillion [COSMETIC]'). Use these observed diffs to justify and drive your minimal EDIT_PLAN. EDIT_PLAN lines MUST include 'TYPE=[STRUCTURAL|COSMETIC]'. When any OBSERVED_DIFFS contains [STRUCTURAL], prioritize STRUCTURAL EDIT_PLANs until those diffs are cleared.\n"
        "\nPLAN AHEAD & AVOID OSCILLATION: After each EDIT_PLAN also include a single short line beginning with 'PREDICTED_SECONDARY_DIFFS:' listing any additional diffs you expect your edit might reveal or leave unresolved (one short sentence). If you predict secondary diffs, include a brief 1-2 step strategy (one short line starting with 'FOLLOWUP_STRATEGY:') describing the next minimal edits you will attempt. Do NOT oscillate between contradictory edits; prefer cumulative, incremental edits that move toward resolving STRUCTURAL diffs.\n"
        "FINAL-OUTPUT FILTERS: If you add pre-window rows solely to compute rolling/lag values, you may propose a minimal final SELECT filter as a last-step STRUCTURAL edit to restrict displayed months/rows so the visible output matches GOLD. Such final filters must be clearly labeled in EDIT_PLAN and justified in OBSERVED_DIFFS (explain which rows are extra and why).\n"
        "PRIORITIZATION NOTE: Always restore non-NULLs and correct join/grouping logic first. Cosmetic edits (rounding, formatting) are allowed only after no [STRUCTURAL] items remain in OBSERVED_DIFFS.\n"
        "COLUMN NAME DIFFERENCES ARE SAFE TO IGNORE: If the only difference between SQL_RESULT and GOLD is the column header name (e.g., 'avg_balance_quadrillions' vs 'average_balance_trillion', or casing differences), treat this as [COSMETIC] and IGNORE it. Column name differences do not affect correctness - only the actual data values, row counts, and row content matter. Do NOT propose edits to change column names unless explicitly required by the query logic.\n"
        "SPECIAL FOCUS ON NULLS: If any OBSERVED_DIFFS reports NULLS for columns that are non-NULL in GOLD, treat these NULLS as HIGH-PRIORITY STRUCTURAL issues. Diagnose likely causes (wrong JOIN type, missing join condition, premature WHERE/filtering, or incorrect grouping) and propose STRUCTURAL EDIT_PLANs that restore non-NULL values before proposing any cosmetic fixes like rounding or formatting. Always report which rows/keys exhibit the NULLs in OBSERVED_DIFFS so the harness can run targeted checks.\n"
        "DO NOT HARDCODE OUTPUTS: Under NO CIRCUMSTANCES propose edits that fabricate final result rows by hardcoding literal SELECTs, VALUES lists, or UNIONs of constants to match GOLD. Such edits are forbidden and will be rejected by the harness.\n"
        "NO TARGETED OUTPUT-FITTING FILTERS: Do NOT propose final WHERE filters that directly target GOLD values (for example: WHERE output='2001' or WHERE month_year IN ('09-2018', '10-2018')). Final filters are allowed ONLY when the rows were added solely to supply pre-window data for rolling/lag calculations; such filters must be explicitly justified in OBSERVED_DIFFS, labeled TYPE=STRUCTURAL, and accompanied by a FOLLOWUP_STRATEGY that shows why they are safe.\n"
        "\nNon-interactive diagnosis mode: if asked, produce one line per CTE: 'CTE <name>: <short sentence> [confidence]'. If none, reply with 'NO_DIFF'.\n"
        "Only report material semantic differences (missing/wrong joins, partitioning/frame, misplaced WHERE, wrong aggregation/denominator, NULL handling, correlated-subquery scope). Do NOT report stylistic/formatting/rounding differences.\n"
    )
    
    # Add engine-specific syntax rules if needed
    engine_syntax = ""
    if engine == "snowflake":
        engine_syntax = (
            "\n\nSNOWFLAKE-SPECIFIC SYNTAX RULES (CRITICAL - You MUST follow these when editing Snowflake SQL):\n"
            "- ALWAYS use fully-qualified three-part names: DATABASE.SCHEMA.TABLE_NAME\n"
            "- NEVER use shortcuts like SCHEMA.TABLE or just TABLE; Snowflake will reject these queries.\n"
            "- Example correct syntax: SELECT * FROM PATENTS.PATENTS.PUBLICATIONS LIMIT 5;\n"
            "- Example WRONG syntax: SELECT * FROM PATENTS.PUBLICATIONS (will fail)\n"
            "\n"
            "COLUMN NAME CASE SENSITIVITY (CRITICAL - READ CAREFULLY):\n"
            "- Snowflake converts unquoted identifiers to UPPERCASE automatically\n"
            "- If a column was created with lowercase (e.g., \"publication_number\"), you MUST quote it: \"publication_number\"\n"
            "- ALWAYS use double quotes around column names to preserve exact case: SELECT \"column_name\", \"another_col\" FROM ...\n"
            "- Example correct: SELECT \"publication_number\", \"country_code\" FROM PATENTS.PATENTS.PUBLICATIONS LIMIT 5;\n"
            "- Example WRONG: SELECT publication_number FROM ... (Snowflake looks for PUBLICATION_NUMBER which doesn't exist)\n"
            "- When making edits, preserve the exact case and quoting style of columns as they appear in the AGENT SQL or GOLD SQL\n"
            "\n"
            "CRITICAL: FLATTEN AND VARIANT COLUMN QUOTING:\n"
            "- When using LATERAL FLATTEN(INPUT => column_name), if the column name is case-sensitive, you MUST quote it: LATERAL FLATTEN(INPUT => \"inputs\")\n"
            "- Example WRONG: LATERAL FLATTEN(INPUT => inputs) → Snowflake looks for INPUTS (uppercase) which may not exist\n"
            "- Example CORRECT: LATERAL FLATTEN(INPUT => \"inputs\") → Uses exact case\n"
            "- When referencing VARIANT fields in WHERE clauses (e.g., block_timestamp), you MUST quote the column name: WHERE TO_TIMESTAMP(\"block_timestamp\" / 1000000)\n"
            "- Example WRONG: WHERE TO_TIMESTAMP(block_timestamp / 1000000) → Error: invalid identifier 'BLOCK_TIMESTAMP'\n"
            "- Example CORRECT: WHERE TO_TIMESTAMP(\"block_timestamp\" / 1000000) → Uses exact case\n"
            "- ALWAYS check the GOLD SQL to see which columns are quoted - match that exact quoting pattern\n"
            "- If GOLD SQL uses \"inputs\", \"outputs\", \"block_timestamp\" (quoted), your edits MUST also quote them\n"
            "\n"
            "Other Snowflake features to preserve in edits:\n"
            "- Array/object access: Use bracket notation and FLATTEN for nested data (e.g., column[0], FLATTEN(input => array_column))\n"
            "- String operations: ILIKE (case-insensitive), CONTAINS, REGEXP patterns\n"
            "- Date/time: Use TO_DATE, TO_TIMESTAMP, DATEADD, DATEDIFF, DATE_TRUNC (not strftime)\n"
            "- Window functions: QUALIFY clause is available for filtering window results\n"
            "- Semi-structured data: VARIANT, ARRAY, OBJECT types; use :: for casting (e.g., column::STRING, not CAST)\n"
            "  * VARIANT columns can be used directly in expressions without casting if the data type is valid\n"
            "  * Use :: to cast VARIANT to specific types: variant_col::FLOAT, variant_col::VARCHAR, variant_col::DATE\n"
            "  * Access nested VARIANT fields: variant_col:field_name or variant_col['field_name'] (no quotes around field names in path)\n"
            "  * Example: SELECT data:name::VARCHAR, data:count::INTEGER FROM DATABASE.SCHEMA.TABLE_NAME;\n"
            "\n"
            "WHEN EDITING SNOWFLAKE SQL:\n"
            "- Preserve all three-part table names (DATABASE.SCHEMA.TABLE) exactly as they appear\n"
            "- Preserve column quoting (use \"column_name\" if columns are quoted in original, use unquoted if they're uppercase)\n"
            "- Do NOT change table names to two-part or one-part formats\n"
            "- Do NOT remove quotes from lowercase column names\n"
            "- Do NOT add quotes to uppercase column names unnecessarily\n"
        )
    elif engine == "bq" or engine == "bigquery":
        engine_syntax = (
            "\n\n" + "="*80 + "\n"
            "BIGQUERY-SPECIFIC SYNTAX RULES (CRITICAL - You MUST follow these when editing BigQuery SQL):\n"
            "="*80 + "\n"
            "- ALWAYS use fully-qualified three-part names: DATABASE.SCHEMA.TABLE\n"
            "- Use backticks for table names: `project_id.dataset_id.table_name`\n"
            "- Example correct syntax: SELECT * FROM `project_id.dataset_id.table_name` LIMIT 5;\n"
            "- NEVER use shortcuts like SCHEMA.TABLE or just TABLE; BigQuery requires fully-qualified names\n"
            "\n"
            "WILDCARD TABLE SUPPORT:\n"
            "- When performing a UNION operation on many tables with similar prefix, you can use a wildcard table to simplify your query\n"
            "- Example: SELECT col1, col2 FROM `project_id.dataset_id.table_prefix*`\n"
            "- Avoid manually listing tables unless absolutely necessary\n"
            "- Use wildcard tables to combine multiple tables with the same schema efficiently\n"
            "\n"
            "CRITICAL: BIGQUERY CTE REQUIREMENTS:\n"
            "- If your edit references any CTE (e.g., FROM visitor_monthly_pageviews, FROM cte1, etc.), you MUST include the complete query with ALL CTE definitions\n"
            "- BigQuery cannot execute fragments that reference CTEs without their definitions\n"
            "- When your edit references a CTE, provide the FULL query: WITH <all_ctes> ... SELECT ... FROM <cte_name>\n"
            "- Example: If editing the final SELECT that uses CTE 'visitor_monthly_pageviews', include: 'WITH visitor_monthly_pageviews AS (...), ... SELECT ... FROM visitor_monthly_pageviews'\n"
            "- Do NOT propose minimal fragments like 'SELECT ... FROM visitor_monthly_pageviews' without the WITH clause - this will fail in BigQuery\n"
            "- Extract the necessary CTEs from the AGENT FULL SQL and include them in your edit\n"
            "\n"
            "\n*** CRITICAL: BIGQUERY STRUCT/COLUMN NAMING IN CTEs - READ THIS CAREFULLY ***\n"
            "RULE 1: When you SELECT totals.pageviews FROM a table, the column in the CTE is named 'pageviews' (NOT 'totals.pageviews')\n"
            "RULE 2: In ANY CTE that has 'FROM <cte_name>' (references another CTE), you MUST use simple column names WITHOUT struct prefixes\n"
            "\nWRONG EXAMPLES (will cause 'Unrecognized name: totals' error):\n"
            "  - SELECT totals.pageviews FROM session_classification ❌\n"
            "  - WHERE totals.pageviews IS NOT NULL (in a CTE referencing another CTE) ❌\n"
            "  - CASE WHEN totals.transactions >= 1 ... (when transactions came from a previous CTE) ❌\n"
            "  - SUM(totals.pageviews) FROM classified_sessions ❌\n"
            "\nCORRECT EXAMPLES:\n"
            "  - SELECT pageviews FROM session_classification ✅\n"
            "  - WHERE pageviews IS NOT NULL ✅\n"
            "  - CASE WHEN transactions >= 1 ... ✅\n"
            "  - SUM(pageviews) FROM classified_sessions ✅\n"
            "\nREMEMBER:\n"
            "- Struct prefix (totals.) is ONLY used when selecting FROM the original table (e.g., FROM `bigquery-public-data...ga_sessions_*`)\n"
            "- Struct prefix is NEVER used when selecting FROM a CTE (e.g., FROM session_classification)\n"
            "- If you see ANY 'Unrecognized name: totals' error, you have a struct prefix in a CTE - find it and remove the prefix\n"
            "- Check ALL CTEs: if a CTE has 'FROM <cte_name>', then ALL column references in that CTE must use simple names\n"
            "\n"
            "WHEN EDITING BIGQUERY SQL:\n"
            "- Preserve all three-part table names (DATABASE.SCHEMA.TABLE) exactly as they appear\n"
            "- Use backticks for table names consistently\n"
            "- Prefer wildcard tables over manual UNION operations when applicable\n"
            "- Do NOT change table names to two-part or one-part formats\n"
            "- If your edit uses CTEs: ALWAYS include complete WITH clause with all CTE definitions\n"
            "- REMEMBER: In CTEs that reference other CTEs, NEVER use struct prefixes (totals., hits., etc.) - use simple column names only\n"
        )
    
    # Build final prompt with inputs
    hint_section = ""
    if hint:
        hint_section = f"\n\n[HINT FOR THIS SPECIFIC INSTANCE]:\n{hint}\n"
    
    prompt = base_prompt + engine_syntax + hint_section + "\n\nInputs:\n"
    
    # Inject schema context if available (for engines that support it)
    if schema_context:
        engine_label = "Snowflake" if engine == "snowflake" else "BigQuery" if (engine == "bq" or engine == "bigquery") else "database"
        prompt += f"- [SCHEMA_CONTEXT] ({engine_label} database schema):\n" + schema_context.strip() + "\n\n"
    
    # Inject external knowledge if available
    if external_knowledge:
        prompt += f"- [EXTERNAL_KNOWLEDGE]:\n" + external_knowledge.strip() + "\n\n"
    
    prompt += (
        "- USER QUERY:\n" + (user_query or "") + "\n\n"
        "- AGENT FULL SQL (all CTEs):\n" + (agent_full_sql_text or "") + "\n\n"
        "- GOLD SQL (authoritative, all CTEs):\n" + (gold_sql_text or "") + "\n\n"
        "- AGENT EXECUTION RESULT (CSV header+rows, optional):\n" + (agent_result_csv_text[:] if agent_result_csv_text else "<none>") + "\n\n"
        "- GOLD EXECUTION RESULT (CSV header+rows):\n" + (gold_result_csv_text[:] if gold_result_csv_text else "") + "\n\n"
        + ("- TRACE (optional):\n" + (processed_trace_text or "") + "\n\n" if processed_trace_text else "")
        + "IMPORTANT: The GOLD SQL above shows the CORRECT approach that produces the matching results.\n"
        + "IMPORTANT: CAREFULLY EXAMINE the GOLD SQL structure (CTEs, joins, filters, date calculations, etc.) for hard to find issues.\n"
        + "CRITICAL: Return PLAIN TEXT only. Follow the interactive protocol exactly. Do NOT output JSON or extra commentary."
    )
    def _extract_tagged(text: str, tag: str) -> Optional[str]:
        m = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    # Build system message based on engine
    if engine == "snowflake":
        system_message = (
            "You are an SQL analyst and MINIMAL-REPAIR agent working with a Snowflake cloud database. "
            "You are repairing SQL queries to match expected results. "
            "When proposing edits, you MUST preserve Snowflake-specific syntax: "
            "fully-qualified three-part table names (DATABASE.SCHEMA.TABLE), "
            "proper column quoting for case-sensitive identifiers, "
            "and Snowflake-specific functions (TO_DATE, DATEADD, :: casting, etc.). "
            "Use <think>..</think> for your reasoning and <sql>..</sql> for SQL edits."
        )
    elif engine == "bq" or engine == "bigquery":
        system_message = (
            "You are an SQL analyst and MINIMAL-REPAIR agent working with a BigQuery cloud database. "
            "You are repairing SQL queries to match expected results. "
            "When proposing edits, you MUST preserve BigQuery-specific syntax: "
            "fully-qualified three-part table names (DATABASE.SCHEMA.TABLE), "
            "backticks for table names when needed, "
            "and prefer wildcard tables over manual UNION operations when combining similar tables. "
            "Use <think>..</think> for your reasoning and <sql>..</sql> for SQL edits."
        )
    else:
        system_message = (
            "You are an SQL analyst and MINIMAL-REPAIR agent. "
            "You are repairing SQL queries to match expected results. "
            "Use <think>..</think> for your reasoning and <sql>..</sql> for SQL edits."
        )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    if verbose:
        print("\n=== STARTING INTERACTIVE DIFF/REPAIR ===")
        print(f"instance_id={instance_id} | max_turns={max_turns} | model={model}")
        # log provider info
        try:
            log_provider(model)
        except Exception:
            pass
        print("--- GOLD SQL (truncated) ---")
        print((gold_sql_text or "")[:2000])
        print("--- AGENT SQL (truncated) ---")
        print((agent_full_sql_text or "")[:2000])
        print("--- GOLD RESULT (truncated) ---")
        print(((gold_result_csv_text or "")[:2000]))
        if agent_result_csv_text:
            print("--- AGENT RESULT (truncated) ---")
            print((agent_result_csv_text or "")[:2000])

    sql_attempts = 0
    executor = None  # Store executor for cleanup if needed

    def _truncate_messages_if_needed(msgs: List[Dict[str, str]], max_message_count: int = 20) -> List[Dict[str, str]]:
        """Truncate old messages to keep conversation manageable. Keep system message and initial prompt, plus most recent messages."""
        if len(msgs) <= max_message_count:
            return msgs
        # Keep system message (first) + initial user prompt (second) + last (max_message_count-2) messages
        result = msgs[:2]  # system + initial prompt
        result.extend(msgs[-(max_message_count-2):])  # most recent messages
        if verbose:
            print(f"[harness] Truncated message history: keeping {len(result)} of {len(msgs)} messages")
        return result
    
    def _is_context_window_error(exception: Exception) -> bool:
        """Check if exception is a context window error from litellm."""
        error_str = str(exception).lower()
        error_type = type(exception).__name__
        # Check for common context window error indicators
        context_keywords = [
            "context window",
            "context_window",
            "token limit",
            "token_limit",
            "too long",
            "maximum context length",
            "context_length_exceeded",
            "request too large",
            "maximum tokens",
        ]
        return any(keyword in error_str for keyword in context_keywords) or "ContextWindowExceededError" in error_type
    
    def _truncate_message_content(msg: Dict[str, str], max_chars: int = 50000) -> Dict[str, str]:
        """Truncate content of a message if it exceeds max_chars. Keeps beginning and end."""
        if "content" not in msg:
            return msg
        content = msg["content"]
        if len(content) <= max_chars:
            return msg
        # Keep first 40% and last 40% with a separator
        keep_start = int(max_chars * 0.4)
        keep_end = int(max_chars * 0.4)
        truncated = content[:keep_start] + "\n\n[... TRUNCATED DUE TO SIZE ...]\n\n" + content[-keep_end:]
        return {**msg, "content": truncated}
    
    def _truncate_messages_content(msgs: List[Dict[str, str]], max_chars_per_msg: int = 50000) -> List[Dict[str, str]]:
        """Truncate content within messages if they're too large. Prioritizes truncating the initial prompt."""
        if not msgs:
            return msgs
        result = []
        for i, msg in enumerate(msgs):
            # For the initial user prompt (index 1), truncate more aggressively
            if i == 1 and "role" in msg and msg["role"] == "user":
                truncated_msg = _truncate_message_content(msg, max_chars=max_chars_per_msg // 2)
            else:
                truncated_msg = _truncate_message_content(msg, max_chars=max_chars_per_msg)
            result.append(truncated_msg)
        return result

    def _detect_static_output_sql(s: str) -> bool:
        """Heuristic detector for SQL that fabricates output rows with literals.
        Returns True if SQL likely hardcodes final output (VALUES, UNION of literals, or SELECT of only literals).
        """
        if not s:
            return False
        low = s.lower()
        # obvious patterns
        if " values (" in low:
            return True
        if re.search(r"union\s+all\s+select\s+(?:null|\d+|'.*?')", low, flags=re.I):
            return True
        # SELECT of only literals (e.g., SELECT 0 AS x, SELECT 'a' AS b)
        if re.search(r"select\s+(?:null|\d+|'.*?')(?:\s+as\s+\w+)?\s*(?:,|$)", low, flags=re.I):
            return True
        return False

    # Track if we've injected the GOLD SQL analysis guidance (after turn 5)
    gold_analysis_injected = False

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n----- TURN {turn}/{max_turns} -----")
        
        # Truncate message history if it's getting too long (prevent context overflow)
        messages = _truncate_messages_if_needed(messages, max_message_count=20)
        
        # After turn 5, if we're still not matching, inject enhanced GOLD SQL analysis guidance
        if turn == 8 and not gold_analysis_injected:
            gold_analysis_injected = True
            if verbose:
                print(f"[harness] Injecting enhanced GOLD SQL analysis guidance after {turn} turns without match")
            
            analysis_prompt = (
                "=" * 80 + "\n"
                "CRITICAL: DEEP GOLD SQL ANALYSIS REQUIRED\n"
                "=" * 80 + "\n"
                "You have tried 5 turns and the SQL_RESULT still doesn't match GOLD. You MUST now provide a DETAILED ANALYSIS.\n"
                "\n"
                "REQUIRED OUTPUT FORMAT in your <think> block:\n"
                "\n"
                "1. GOLD SQL ANALYSIS:\n"
                "   - What does the GOLD SQL do step-by-step?\n"
                "   - What CTEs does it create and what does each CTE compute?\n"
                "   - What joins does it perform (type, tables, join conditions)?\n"
                "   - What filters/WHERE conditions does it apply?\n"
                "   - What date calculations, transformations, or special logic does it use?\n"
                "   - What columns does it select and how are they named?\n"
                "\n"
                "2. AGENT SQL ANALYSIS:\n"
                "   - What does your AGENT SQL do step-by-step?\n"
                "   - What CTEs does it create (if any) and what do they compute?\n"
                "   - What joins does it perform (type, tables, join conditions)?\n"
                "   - What filters/WHERE conditions does it apply?\n"
                "   - What date calculations, transformations, or special logic does it use?\n"
                "   - What columns does it select and how are they named?\n"
                "\n"
                "3. DETAILED COMPARISON - WHAT IS WRONG:\n"
                "   - Point-by-point differences between GOLD SQL and AGENT SQL:\n"
                "     * CTE differences (missing CTEs, different CTE logic, different CTE structure)\n"
                "     * Join differences (join type, join conditions, missing joins, extra joins)\n"
                "     * Filter differences (WHERE clause conditions, missing filters, extra filters)\n"
                "     * Column name casing differences (e.g., 'COUNTRY' vs 'country' vs 'Country')\n"
                "     * Table/schema name casing differences (e.g., 'STANDARD_TILE' vs 'standard_tile')\n"
                "     * Date calculation differences (how dates are computed, reference dates, date ranges)\n"
                "     * Column selection differences (missing columns, extra columns, aliases)\n"
                "   - Identify which of these differences likely causes empty results or mismatches\n"
                "   - If you copied GOLD SQL structure but still got empty results, explain why:\n"
                "     * Column name mismatches in NATURAL JOIN (must match exactly including casing)\n"
                "     * Table/schema name casing mismatches\n"
                "     * Subtle differences in date calculations, filters, or join conditions\n"
                "\n"
                "4. PROPOSED FIX:\n"
                "   - Based on your detailed analysis above, propose MINIMAL edits\n"
                "   - Explain which specific differences you will fix and why\n"
                "   - Then provide your SQL edit in <sql>...</sql> block\n"
                "\n"
                "REFERENCE - YOUR AGENT SQL:\n" + ("-" * 80 + "\n") + (agent_full_sql_text or "") + "\n" + ("-" * 80 + "\n\n")
                + "REFERENCE - GOLD SQL (CORRECT APPROACH):\n" + ("-" * 80 + "\n") + (gold_sql_text or "") + "\n" + ("-" * 80 + "\n\n")
                + "Now provide the detailed analysis (GOLD SQL ANALYSIS, AGENT SQL ANALYSIS, WHAT IS WRONG) in your <think> block,\n"
                + "then propose your fix.\n"
                + "=" * 80 + "\n"
            )
            messages.append({"role": "user", "content": analysis_prompt})
        
        # Call LLM raw so we can capture reasoning_content when present
        # Handle context window errors with truncation and retry
        resp = None
        context_window_error = False
        try:
            # import pdb; pdb.set_trace()
            resp = litellm.completion(model=model, messages=messages)
        except Exception as e:
            if _is_context_window_error(e):
                context_window_error = True
                if verbose:
                    print(f"[harness] Context window error detected: {e}")
                    print(f"[harness] Attempting to recover by truncating message content (current: {len(messages)} messages)")
                
                # First, try truncating CONTENT within messages (especially initial prompt)
                truncated_messages = _truncate_messages_content(messages, max_chars_per_msg=30000)
                if verbose:
                    original_sizes = [len(m.get("content", "")) for m in messages]
                    new_sizes = [len(m.get("content", "")) for m in truncated_messages]
                    print(f"[harness] Truncated message content: {sum(original_sizes)} -> {sum(new_sizes)} total chars")
                
                # Also reduce message count if still many messages
                if len(truncated_messages) > 7:
                    truncated_messages = truncated_messages[:2] + truncated_messages[-(5):]
                    if verbose:
                        print(f"[harness] Also reduced to {len(truncated_messages)} messages (system + initial + last 5)")
                
                # Retry with truncated content (one retry only)
                try:
                    resp = litellm.completion(model=model, messages=truncated_messages)
                    messages = truncated_messages  # Update messages for rest of the turn
                    if verbose:
                        print(f"[harness] Retry with truncated content succeeded")
                except Exception as retry_error:
                    # Retry also failed - raise the original error or a clean context window error
                    if verbose:
                        print(f"[harness] Retry with truncated content also failed: {retry_error}")
                    # Raise a clean exception that will be caught by run_diff_for_instance
                    raise Exception(f"CONTEXT_WINDOW_EXCEEDED: Unable to recover from context window error even after content truncation. Original error: {str(e)[:200]}")
            else:
                # Not a context window error, re-raise as-is
                raise
        
        # import pdb; pdb.set_trace()
        try:
            msg_obj = resp["choices"][0]["message"]
        except Exception:
            try:
                msg_obj = resp.choices[0].message
            except Exception:
                msg_obj = None
        # Extract content preferring explicit content, else reasoning_content
        content = None
        try:
            if isinstance(msg_obj, dict):
                content = msg_obj.get("content") or msg_obj.get("reasoning_content")
            else:
                content = getattr(msg_obj, "content", None) or getattr(msg_obj, "reasoning_content", None)
        except Exception:
            content = None
        text = (content or "").strip()
        if not text:
            if verbose:
                print("[LLM] no response")
            break
        if verbose:
            print("[assistant]")
            if text:
                print(text if len(text) < 2000 else text[:2000] + "...")
            # if reasoning content separately available, print it
            try:
                reasoning = (msg_obj.get("reasoning_content") if isinstance(msg_obj, dict) else getattr(msg_obj, "reasoning_content", None))
                if reasoning and reasoning.strip() and reasoning.strip() != text:
                    print("[assistant reasoning]")
                    print(reasoning if len(reasoning) < 2000 else reasoning[:2000] + "...")
            except Exception:
                pass

        # Detect immediate final signal in assistant text
        if "MATCH_OK" in text or "MINIMAL_FIX" in text:
            if verbose:
                print("[assistant declared MATCH_OK / provided MINIMAL_FIX]")
            # Collect EDIT_PLANs from prior assistant messages
            edits = []
            for m in messages:
                if m.get("role") == "assistant":
                    c = m.get("content") or ""
                    # look for EDIT_PLAN: lines anywhere in the content
                    for ed in re.findall(r"EDIT_PLAN\s*:\s*(.*)", c, flags=re.IGNORECASE):
                        e = ed.strip()
                        if e and e not in edits:
                            edits.append(e)
            if verbose:
                print("[harness] collected EDIT_PLANs:")
                for i, e in enumerate(edits, 1):
                    print(f"  {i}) {e}")

            # Ask assistant to produce cumulative list and mark necessary vs cosmetic and then output minimal required edits
            observed_edits_text = "\n".join([f"{i}) {e}" for i, e in enumerate(edits, 1)]) if edits else "(no edits recorded)"
            
            ask = (
                "You applied these EDIT_PLANs during the session:\n"
                + observed_edits_text
                + "\n\n"
                + "IMPORTANT: Review the entire conversation history above. You have access to:\n"
                + "- The ORIGINAL AGENT SQL (from the initial prompt)\n"
                + "- All the SQL you proposed in <sql> blocks during each turn\n"
                + "- The SQL_RESULT responses showing what worked\n"
                + "- The FINAL SQL that produced the matching result\n\n"
                + "For EACH of the EDIT_PLANs above: mark it [NECESSARY] or [COSMETIC].\n"
                + "CRITICAL: Only mark an edit as [NECESSARY] if:\n"
                + "  1) It was actually applied in the FINAL SQL that matched GOLD (check the last <sql> block you proposed that got SQL_RESULT matching GOLD)\n"
                + "  2) You can see the actual SQL change between ORIGINAL and FINAL SQL in the conversation history\n"
                + "If an EDIT_PLAN was mentioned but NOT visible in the final working SQL, mark it as [COSMETIC] or exclude it from MINIMAL_REQUIRED_EDITS.\n"
                + "Important: Any edit that changed a CTE definition, JOIN condition, or CTE-level predicate that appears in the final SQL should be marked [NECESSARY].\n"
                + "Then output the MINIMAL_REQUIRED_EDITS list (only the edits that are actually visible in the SQL differences between ORIGINAL and the FINAL matching SQL).\n"
                + "Output format (plain text only):\nCUMULATIVE_EDITS:\n1) <edit text> — [NECESSARY|COSMETIC]\n...\nMINIMAL_REQUIRED_EDITS:\n- <edit1> (must match what's actually in final SQL)\n- <edit2> (must match what's actually in final SQL)\nEVIDENCE: one short line summarizing which rows/columns now match.\n"
                + "Also output a CLEAN_SUMMARY summarizing the core ISSUE and the EDIT that addressed it (must describe actual SQL changes visible in conversation).\n"
                + "Do NOT include any other commentary."
            )
            messages.append({"role": "user", "content": ask})
            final = _llm(model, messages)
            if verbose:
                print("[assistant final cumulative edits]")
                print(final)
            
            # Cleanup: close executor if it has a close method (important for Snowflake persistent connections)
            if executor is not None and hasattr(executor, 'close'):
                try:
                    executor.close()
                except Exception:
                    pass
            
            return (final or "").strip()

        thinking = _extract_tagged(text, "think")
        sql_block = _extract_tagged(text, "sql")
        # Accept <solution> as a fallback/full-SQL tag produced by some assistants
        solution_block = _extract_tagged(text, "solution")
        if not sql_block and solution_block:
            sql_block = solution_block
            if verbose:
                print("[harness] detected <solution> block; treating as SQL to execute")
        
        # # Fallback: if no tagged SQL but text looks like SQL (contains SELECT/INSERT/UPDATE/DELETE/WITH and SQL keywords), treat entire text as SQL
        # if not sql_block and text:
        #     # Check if text looks like SQL (contains common SQL keywords)
        #     sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"]
        #     text_upper = text.upper()
        #     if any(keyword in text_upper for keyword in sql_keywords):
        #         # If text contains SQL keywords but no other protocol elements (like EDIT_PLAN), treat as raw SQL
        #         if "EDIT_PLAN" not in text.upper() and "<think>" not in text.lower():
        #             sql_block = text.strip()
        #             if verbose:
        #                 print("[harness] detected untagged SQL; treating entire response as SQL to execute")
        
        # Append assistant content (use raw text)
        messages.append({"role": "assistant", "content": text})

        if sql_block:
            # proceed to execute SQL (we rely on the prompt to require an EDIT_PLAN before SQL)
            sql_attempts += 1
            if sql_attempts > max_turns:
                if verbose:
                    print("Reached maximum SQL attempts; stopping interactive repair.")
                break
            sql_to_run = sql_block.strip()
            
            # VALIDATION: Check if SQL is too similar to GOLD and not to AGENT (copying entire GOLD SQL)
            def _compute_sql_similarity(sql1: str, sql2: str) -> float:
                """Compute simple similarity between two SQL queries (normalized token overlap)."""
                if not sql1 or not sql2:
                    return 0.0
                # Normalize: lowercase, remove extra whitespace
                s1 = ' '.join(sql1.lower().split())
                s2 = ' '.join(sql2.lower().split())
                # Tokenize by splitting on whitespace and punctuation
                import re
                tokens1 = set(re.findall(r'\w+', s1))
                tokens2 = set(re.findall(r'\w+', s2))
                if not tokens1 or not tokens2:
                    return 0.0
                # Jaccard similarity
                intersection = len(tokens1 & tokens2)
                union = len(tokens1 | tokens2)
                return intersection / union if union > 0 else 0.0
            
            agent_similarity = _compute_sql_similarity(sql_to_run, agent_full_sql_text or "")
            gold_similarity = _compute_sql_similarity(sql_to_run, gold_sql_text or "")
            
            # If SQL is more similar to GOLD than to AGENT (and both similarities are high), it's likely a copy
            # Threshold: if gold_similarity > 0.7 and gold_similarity > agent_similarity + 0.2, reject
            if gold_similarity > 0.7 and gold_similarity > agent_similarity + 0.2:
                sql_result_text = (
                    f"SQL_REJECTED_MINIMALITY_VIOLATION: The proposed SQL is too similar to GOLD SQL "
                    f"(similarity: {gold_similarity:.2f}) and not similar enough to AGENT SQL "
                    f"(similarity: {agent_similarity:.2f}). "
                    f"You MUST start from AGENT SQL and make MINIMAL incremental edits, not copy the entire GOLD SQL structure. "
                    f"Please identify the SPECIFIC clause(s) in AGENT SQL that need fixing and modify only those."
                )
                if verbose:
                    print(f"[harness] Rejected assistant SQL: minimality violation (GOLD similarity: {gold_similarity:.2f}, AGENT similarity: {agent_similarity:.2f})")
                messages.append({"role": "user", "content": sql_result_text})
                continue
            
            if verbose:
                print("[Executing SQL proposed by assistant]")
                print(sql_to_run if len(sql_to_run) < 2000 else sql_to_run[:2000] + "...")
            sql_result_text = "<no_db>"
            # Reject SQLs that attempt to hardcode final outputs
            if False:#_detect_static_output_sql(sql_to_run):
                sql_result_text = "SQL_REJECTED_STATIC_OUTPUT: assistant attempted to hardcode output rows (VALUES/UNION/literal SELECT) - forbidden"
                if verbose:
                    print("[harness] Rejected assistant SQL: static-output detected")
            else:
                if db_path_or_cred:
                    try:
                        # Create executor using factory pattern (supports SQLite, Snowflake, BigQuery)
                        # Reuse executor for all queries in the loop (important for Snowflake persistent connections)
                        if executor is None:
                            executor = make_executor(engine, db_path_or_cred)
                        headers, rows = executor.execute(sql_to_run)
                        lines = []
                        if headers:
                            lines.append(",".join([str(h) for h in headers]))
                        for r in rows:
                            lines.append(",".join([str(x) if x is not None else "" for x in r]))
                        # Truncate long results aggressively to avoid context window issues
                        total_rows = len(rows)
                        max_display = 50  # Reduced from 200 to 50 to prevent context overflow
                        if total_rows > max_display:
                            # keep header + first max_display rows
                            display_lines = [lines[0]] + lines[1:1+max_display] if headers else lines[:max_display]
                            sql_result_text = "\n".join(display_lines)
                            sql_result_text += f"\n... TRUNCATED: showing {min(max_display, total_rows)} of {total_rows} rows"
                            if verbose:
                                print(f"[harness] SQL_RESULT truncated: {total_rows} total rows, showing {min(max_display, total_rows)}")
                        else:
                            sql_result_text = "\n".join(lines)
                    except Exception as e:
                        sql_result_text = f"SQL_ERROR: {e}"
            if verbose:
                print("[SQL_RESULT returned]")
                print(sql_result_text if len(sql_result_text) < 2000 else sql_result_text[:2000] + "...")
            messages.append({"role": "user", "content": "SQL_RESULT:\n" + sql_result_text})
            continue

        # No sql_block: check if it's NO_DIFF or if we should continue
        if "NO_DIFF" in (text or ""):
            # Validate: if there's a clear structural difference, reject NO_DIFF
            # Check row counts: if SQL_RESULT is empty and GOLD has rows (or vice versa), reject NO_DIFF
            def _count_rows(csv_text: Optional[str]) -> int:
                """Count data rows (excluding header) in CSV-like text"""
                if not csv_text or csv_text.strip() == "":
                    return 0
                lines = csv_text.strip().split('\n')
                # Skip header if present (first line)
                # If only header or empty, return 0
                if len(lines) <= 1:
                    return 0
                # Count non-empty data lines (exclude header)
                return len([l for l in lines[1:] if l.strip()])
            
            gold_row_count = _count_rows(gold_result_csv_text) if gold_result_csv_text else 0
            # Get latest SQL_RESULT from messages if available
            latest_sql_result = None
            for m in reversed(messages):
                if m.get("role") == "user" and "SQL_RESULT" in (m.get("content") or ""):
                    latest_sql_result = m.get("content", "")
                    break
            
            sql_result_row_count = None
            if latest_sql_result:
                # Extract the actual CSV content from "SQL_RESULT:\n<content>"
                sql_result_content = latest_sql_result.split("SQL_RESULT:\n")[-1] if "SQL_RESULT:\n" in latest_sql_result else latest_sql_result
                sql_result_row_count = _count_rows(sql_result_content)
            elif agent_result_csv_text:
                # If no SQL_RESULT yet, check initial agent result
                sql_result_row_count = _count_rows(agent_result_csv_text)
            
            # Row count validation: Check for 0 vs non-zero only (not unequal counts)
            # NOTE: This check can be disabled via environment variable CHECK_UNEQUAL_ROWS=false
            # If disabled, only check for 0 vs non-zero mismatch (one has 0, other has >0)
            check_unequal_rows = os.environ.get("CHECK_UNEQUAL_ROWS", "false").lower() == "true"
            
            # Always reject NO_DIFF if one result has 0 rows and the other has >0 rows (clear structural difference)
            if sql_result_row_count is not None:
                one_empty = (sql_result_row_count == 0 and gold_row_count > 0) or (sql_result_row_count > 0 and gold_row_count == 0)
                # Optionally also check for unequal row counts (disabled by default)
                unequal_counts = check_unequal_rows and (gold_row_count != sql_result_row_count)
                
                if one_empty or unequal_counts:
                    result_label = "SQL_RESULT" if latest_sql_result else "AGENT RESULT"
                    if one_empty:
                        reason = f"{result_label} has {sql_result_row_count} rows (empty) while GOLD has {gold_row_count} rows (non-empty) - this is a STRUCTURAL DIFFERENCE"
                    else:
                        reason = f"{result_label} has {sql_result_row_count} rows while GOLD has {gold_row_count} rows - this is a STRUCTURAL DIFFERENCE"
                    
                    if verbose:
                        print(f"[harness] Rejecting NO_DIFF: {reason}")
                    # Force the LLM to continue by injecting a correction message
                    messages.append({
                        "role": "user",
                        "content": f"ERROR: You declared NO_DIFF, but there is a clear STRUCTURAL DIFFERENCE: {reason}. You MUST examine the GOLD SQL carefully to understand how it retrieves the data (it may use different tables, joins, date calculations, or filters). Propose SQL edits based on the GOLD SQL approach. DO NOT declare NO_DIFF or MATCH_OK when row counts differ."
                    })
                    continue  # Continue the loop instead of returning
            
            # OLD CODE (disabled - was checking for unequal counts):
            # if sql_result_row_count is not None and gold_row_count != sql_result_row_count:
            #     result_label = "SQL_RESULT" if latest_sql_result else "AGENT RESULT"
            #     if verbose:
            #         print(f"[harness] Rejecting NO_DIFF: {result_label} has {sql_result_row_count} rows, GOLD has {gold_row_count} rows - this is a STRUCTURAL DIFFERENCE")
            #     # Force the LLM to continue by injecting a correction message
            #     messages.append({
            #         "role": "user",
            #         "content": f"ERROR: You declared NO_DIFF, but there is a clear STRUCTURAL DIFFERENCE: {result_label} has {sql_result_row_count} data rows while GOLD has {gold_row_count} data rows. You MUST examine the GOLD SQL carefully to understand how it retrieves the data (it may use different tables, joins, date calculations, or filters). Propose SQL edits based on the GOLD SQL approach. DO NOT declare NO_DIFF or MATCH_OK when row counts differ."
            #     })
            #     continue  # Continue the loop instead of returning
            return "NO_DIFF"
        
        # If no sql_block and not NO_DIFF, the assistant should have provided an EDIT_PLAN
        # but didn't provide SQL. This might mean they're stuck or refusing to work.
        # Inject a message forcing them to either provide SQL or declare NO_DIFF.
        if verbose:
            print("[harness] WARNING: Assistant response contains no <sql> block and is not NO_DIFF")
            print("[harness] Forcing assistant to provide SQL or declare NO_DIFF")
        # Inject a message forcing action
        messages.append({
            "role": "user",
            "content": "ERROR: You must follow the protocol. You MUST provide either:\n1) An EDIT_PLAN followed by a <sql>...</sql> block with your proposed SQL edit, OR\n2) 'NO_DIFF' if you cannot proceed further.\n\nRepeating explanations without SQL edits is not allowed. If your query returns empty but GOLD has data, examine the GOLD SQL carefully and propose a different SQL approach. If you truly cannot proceed, respond with exactly 'NO_DIFF'."
        })
        continue
    # Max turns reached: ask for concise minimal summary with observed diffs
    if verbose:
        print("[harness] MAX_TURNS reached without MATCH_OK")
    ask = (
        "MAX_TURNS_REACHED: Based on all SQL_RESULTs observed, produce PLAIN TEXT only with the following sections:\n"
        "1) OBSERVED_DIFFS: short bullet list of concrete observed differences vs GOLD (row counts, NULLs per column, value scale, missing rows, ordering);\n"
        "2) CUMULATIVE_EDITS: list of EDIT_PLAN entries you applied during the session (numbered) with each marked [NECESSARY] or [COSMETIC];\n"
        "3) MINIMAL_REQUIRED_EDITS: the minimal subset of those edits absolutely required to fix the divergence (one per line);\n"
        "4) CLEAN_SUMMARY: a short summary summarizing the core ISSUE, and the EDIT that addressed it.\n"
        "If none, output NO_DIFF. Do NOT include any other commentary."
    )
    messages.append({"role": "user", "content": ask})
    final = _llm(model, messages)
    result = ("MAX_TURNS_REACHED\n" + (final or "")).strip()
    
    # Cleanup: close executor if it has a close method (important for Snowflake persistent connections)
    if executor is not None and hasattr(executor, 'close'):
        try:
            executor.close()
        except Exception:
            pass
    
    return result


def generate_rules_from_diff(
    diff_text: str,
    agent_sql_text: Optional[str] = None,
    agent_result_csv_text: Optional[str] = None,
    model: str = "azure/o4-mini",
    verbose: bool = False,
) -> Optional[str]:
    """
    Convert a full textual diff/repair session output into concise, actionable
    human-readable rules/checks/hints suitable for storing as memory.

    The returned string is plain text (numbered bullets). Each bullet should
    contain: a short id, a one-line rule/hint, a short check hint (what to look
    for in SQL/results), a suggested minimal fix, and a severity tag.

    This function intentionally returns the LLM-generated text verbatim so the
    caller can inspect and store rules as desired.
    """
    if not diff_text:
        return None
    # "Generic memories should be framed in a way that captured the important lessons for the validator to avoid missing these mistakes for any question in the future. Keep them clear and easy to understand. Avoid hallucinating.\n\n"
    # rule_prompt = (
    #     "You are an expert SQL analyst's memory builder.\n"
    #     "Given the following DIFF/REPAIR session output between an AGENT SQL and the GOLD for a single question, produce ONLY plain-English MEMORIES in TWO labeled sections and NOTHING ELSE. No JSON, no checks, no fixes, no evidence, no metadata, no extra commentary.\n\n"
    #     "CATEGORIZATION RULES:\n"
    #     "- DATABASE_MEMORIES: Database-specific memories focused on specific tables, columns, and operations for THIS database schema. These should reference exact table/column names from the DIFF/SQL when available. Frame them as validation checks for this specific database schema. These memories are reusable across questions on the same database.\n"
    #     "- GENERIC_MEMORIES: General lessons that apply across any database/question. These should abstract away specific table/column names and focus on SQL patterns, join logic, aggregation principles, etc.\n\n"
    #     "Output format (STRICT):\n"
    #     "DATABASE_MEMORIES (one per line):\n"
    #     "Format: VERIFY/CHECK/ENSURE: <specific validation check for the tables/columns mentioned in DIFF> | CONTEXT: <explain what tables/columns are involved and why this check matters for this database> | WHEN_TO_CHECK: <describe when this validation applies - e.g., 'when joining table X with table Y' or 'when aggregating column Z'> | EXAMPLE_USAGE: <detailed example showing: when you have table X and column Y, verify that you do Z (with concrete SQL fragment showing correct vs incorrect approach)>\n"
    #     "Good example: VERIFY: The JOIN between ball_by_ball and batsman_scored includes innings_no matching | CONTEXT: ball_by_ball table contains match_id, over_id, ball_id, innings_no; batsman_scored table contains match_id, over_id, ball_id, innings_no, runs_scored. When these tables are joined, all matching keys must be included to avoid cross-innings data misalignment | WHEN_TO_CHECK: When joining ball_by_ball with batsman_scored to compute batting statistics per match | EXAMPLE_USAGE: When joining these tables, verify the JOIN condition includes all matching keys: 'ON b.match_id = s.match_id AND b.over_id = s.over_id AND b.ball_id = s.ball_id AND b.innings_no = s.innings_no' is correct. Missing the innings_no condition (like 'ON b.match_id = s.match_id AND b.over_id = s.over_id AND b.ball_id = s.ball_id') causes runs from different innings to be incorrectly matched, leading to doubled values.\n"
    #     "...\n\n"
    #     "GENERIC_MEMORIES (one per line):\n"
    #     "Format: VERIFY/CHECK/ENSURE: <general validation principle> | PRINCIPLE: <explain the underlying SQL principle or pattern> | WHEN_TO_APPLY: <describe when this principle applies broadly> | EXAMPLE_USAGE: <detailed example showing: when you encounter situation X (without specific table names), then verify Y (with abstract SQL pattern showing correct vs incorrect approach)>\n"
    #     "Good example: VERIFY: JOIN clauses include all necessary composite keys | PRINCIPLE: When tables have multi-part primary/composite keys (e.g., match_id + innings_no, or order_id + item_id), all key components must be included in JOIN conditions to ensure correct record matching | WHEN_TO_APPLY: Whenever joining tables that have composite or multi-column keys, verify all key components are included in the JOIN ON clause | EXAMPLE_USAGE: When joining two tables with composite keys (e.g., table A has columns (id1, id2) and table B has columns (id1, id2, data)), verify the JOIN includes both: 'ON A.id1 = B.id1 AND A.id2 = B.id2' is correct. Missing one key component (like 'ON A.id1 = B.id1') can cause incorrect cross-matching where rows with same id1 but different id2 get incorrectly joined, leading to duplicated or misaligned data.\n"
    #     "...\n\n"
    #     "IMPORTANT GUIDELINES:\n"
    #     "- Each MEMORY must be grounded in the DIFF/SQL/results provided below\n"
    #     "- DATABASE_MEMORIES should use exact table/column names from the DIFF when available\n"
    #     "- GENERIC_MEMORIES should abstract away specific names and focus on patterns\n"
    #     "- EXAMPLES should be detailed and explanatory (not just one-line SQL fragments)\n"
    #     "- EXAMPLES should clearly show correct vs incorrect approaches\n"
    #     "- Frame all memories as refiner-facing checks (what to verify, not assertions)\n"
    #     "- Avoid vague wording; use concrete table/column names in DATABASE_MEMORIES\n"
    #     "- Do not hallucinate table/column names not present in the DIFF\n\n"
    #     "DIFF:\n" + diff_text + "\n"
    #     "Note: If the DIFF indicates ambiguous/incorrect tables or columns, formatting issues, wrong JOINs, misplaced filters, logical/math errors, or unexpected NULLs, the GENERIC_MEMORIES should record a concise, testable lesson that guides refiners to check for and avoid the same mistake in future queries.\n"
    #     "Include only the relevant and grounded short verification lines for the given DIFF. Do not hallucinate. Do not include others.\n"
    #     "Make sure rules are clear and actionable. Avoid vague wording. Use concrete descriptions and examples.\n"
    #     "Now produce ONLY the DATABASE_MEMORIES and GENERIC_MEMORIES sections for the provided DIFF. Make sure each memory is well-framed with detailed examples that explain when and how to apply the check.\n\n"
    # )
    rule_prompt = (
        "You are an expert SQL analyst's memory builder.\n"
        "Given the following DIFF/REPAIR session output between an AGENT SQL and the GOLD for a single question, produce ONLY plain-English MEMORIES in TWO labeled sections and NOTHING ELSE. No JSON, no checks, no fixes, no evidence, no metadata, no extra commentary.\n\n"
        "CATEGORIZATION RULES:\n"
        "- DATABASE_MEMORIES: Database-specific memories focused on specific tables, columns, and operations for THIS database schema. These should reference exact table/column names from the DIFF/SQL when available. Frame them as validation checks advice or data insights for this specific database schema. These memories are reusable across questions on the same database.\n"
        "- GENERIC_MEMORIES: General lessons that apply across any database/question. These should abstract away specific table/column names and focus on SQL patterns, join logic, aggregation principles, etc.\n\n"
        "Output format (STRICT):\n"
        "DATABASE_MEMORIES (one per line):\n"
        "Format: REMEMBER/VERIFY/ENSURE: <actionable advice for the validation agent for a future question on this database> | CONTEXT: <explain what tables/columns are involved and why this check matters for this database> | WHEN_TO_CHECK: <describe when this validation applies - e.g., 'when joining table X with table Y' or 'when aggregating column Z'> | EXAMPLE_USAGE: <detailed example showing: when you have table X and column Y, verify that you do Z (with concrete SQL fragment showing correct vs incorrect approach)>\n"
        "Good example: ENSURE: when JOINING ball_by_ball and batsman_scored, make sure to include innings_no | CONTEXT: ball_by_ball table contains match_id, over_id, ball_id, innings_no; batsman_scored table contains match_id, over_id, ball_id, innings_no, runs_scored. When these tables are joined, all matching keys must be included to avoid cross-innings data misalignment | WHEN_TO_CHECK: When joining ball_by_ball with batsman_scored | EXAMPLE_USAGE: When joining these tables, verify the JOIN condition includes all matching keys: 'ON b.match_id = s.match_id AND b.over_id = s.over_id AND b.ball_id = s.ball_id AND b.innings_no = s.innings_no' is correct. Missing the innings_no condition (like 'ON b.match_id = s.match_id AND b.over_id = s.over_id AND b.ball_id = s.ball_id') causes runs from different innings to be incorrectly matched, leading to doubled values.\n"
        "Good example: REMEMBER: the data for pre-2000 races are in the results table. This table must be used in conjunction with constructor_standings for full races results data | CONTEXT: ... | WHEN_TO_CHECK: When performing queries about race results on F1 database | EXAMPLE_USAGE: ....\n"
        "...\n\n"
        "GENERIC_MEMORIES (one per line):\n"
        "Format: VERIFY/CHECK/ENSURE: <general validation principle> | PRINCIPLE: <explain the underlying SQL principle or pattern> | WHEN_TO_APPLY: <describe when this principle applies broadly> | EXAMPLE_USAGE: <detailed example showing: when you encounter situation X (without specific table names), then verify Y (with abstract SQL pattern showing correct vs incorrect approach)>\n"
        "Good example: VERIFY: JOIN clauses include all necessary composite keys | PRINCIPLE: When tables have multi-part primary/composite keys (e.g., match_id + innings_no, or order_id + item_id), all key components must be included in JOIN conditions to ensure correct record matching | WHEN_TO_APPLY: Whenever joining tables that have composite or multi-column keys, verify all key components are included in the JOIN ON clause | EXAMPLE_USAGE: When joining two tables with composite keys (e.g., table A has columns (id1, id2) and table B has columns (id1, id2, data)), verify the JOIN includes both: 'ON A.id1 = B.id1 AND A.id2 = B.id2' is correct. Missing one key component (like 'ON A.id1 = B.id1') can cause incorrect cross-matching where rows with same id1 but different id2 get incorrectly joined, leading to duplicated or misaligned data.\n"
        "...\n\n"
        "IMPORTANT GUIDELINES:\n"
        "- Each MEMORY must be grounded in the DIFF/SQL/results provided below\n"
        "- DATABASE_MEMORIES should use exact table/column names from the DIFF when available\n"
        "- GENERIC_MEMORIES should abstract away specific names and focus on patterns\n"
        "- EXAMPLES should be detailed and explanatory (not just one-line SQL fragments)\n"
        "- EXAMPLES should clearly show correct vs incorrect approaches\n"
        "- Frame all memories as refiner-facing checks (what to verify, not assertions)\n"
        "- Avoid vague wording; use concrete table/column names in DATABASE_MEMORIES\n"
        "- Do not hallucinate table/column names not present in the DIFF\n\n"
        "DIFF:\n" + diff_text + "\n"
        "Note: If the DIFF indicates ambiguous/incorrect tables or columns, formatting issues, wrong JOINs, misplaced filters, logical/math errors, or unexpected NULLs, the GENERIC_MEMORIES should record a concise, testable lesson that guides refiners to check for and avoid the same mistake in future queries.\n"
        "Include only the relevant and grounded short verification lines for the given DIFF. Do not hallucinate. Do not include others.\n"
        "Make sure rules are clear and actionable. Avoid vague wording. Use concrete descriptions and examples.\n"
        "Now produce ONLY the DATABASE_MEMORIES and GENERIC_MEMORIES sections for the provided DIFF. Make sure each memory is well-framed with detailed examples that explain when and how to apply the check.\n\n"
    )

    messages = [
        {"role": "system", "content": "You are an expert SQL analyst's memory builder."},
        {"role": "user", "content": rule_prompt},
    ]

    if verbose:
        print("[generate_memories_from_diff] prompt:\n", rule_prompt[:2000])

    out = _llm(model, messages)
    return out.strip() if out else None


def run_diff_for_instance(instance_id: str, outputs_dir: str, jsonl_path: Optional[str] = None, db_path: Optional[str] = None, max_turns: int = 6, model: str = "azure/o4-mini", debug: bool = False, hint: Optional[str] = None) -> None:
    """
    Helper main to run interactive diff/repair for an instance using files under outputs_dir.
    Expects files: execution_query.sql, gt_query.sql, execution_result.csv, gt_result.csv, processed_trace.txt (optional).
    Prints turn logs to terminal.
    
    Args:
        debug: If True, stop after getting diff results and write to debug_result.txt (skip tagging/indexing).
        hint: Optional hint to provide to the model when generating the diff.
    """
    out = Path(outputs_dir)
    # strict: require outputs_dir exists
    if not out.exists():
        raise FileNotFoundError(f"outputs_dir does not exist: {out}")
    # enforce presence of expected files
    required = [out / "execution_query.sql", out / "execution_result.csv", out / "gt_result.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required output files in {out}: {missing}")
    
    agent_sql_path = out / "execution_query.sql"
    # Check for gold SQL file: try gt_query.sql first, then fall back to {instance_id}.sql (for Snowflake pattern)
    gold_sql_path = out / "gt_query.sql"
    if not gold_sql_path.exists():
        gold_sql_path = out / f"{instance_id}.sql"
    if not gold_sql_path.exists():
        raise FileNotFoundError(f"Missing gold SQL file in {out}: tried 'gt_query.sql' and '{instance_id}.sql'")
    agent_result_path = out / "execution_result.csv"
    gold_result_path = out / "gt_result.csv"
    trace_path = out / "processed_trace.txt"



    agent_sql = agent_sql_path.read_text(encoding='utf-8') if agent_sql_path.exists() else ""
    gold_sql = gold_sql_path.read_text(encoding='utf-8') if gold_sql_path.exists() else ""
    agent_result_raw = agent_result_path.read_text(encoding='utf-8') if agent_result_path.exists() else None
    gold_result_raw = gold_result_path.read_text(encoding='utf-8') if gold_result_path.exists() else ""
    trace = trace_path.read_text(encoding='utf-8') if trace_path.exists() else None

    # Format CSV results as Markdown tables so the LLM sees clean headers and rows.
    def _is_csv_like(s: Optional[str]) -> bool:
        if not s:
            return False
        s_str = str(s).strip()
        if s_str.startswith("SQL_ERROR:") or s_str.startswith("SQL_REJECTED_STATIC_OUTPUT:"):
            return False
        # Exclude obvious JSON blobs
        if s_str.startswith("{") or s_str.startswith("["):
            return False
        # If it contains multiple lines or commas, treat as CSV-like
        if ("\n" in s_str) or ("," in s_str):
            return True
        # Allow single numeric values (e.g., "0" or "2001") to be treated as CSV
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", s_str):
            return True
        return False

    agent_result = None
    if _is_csv_like(agent_result_raw):
        agent_result = format_csv_as_table(agent_result_raw)
    else:
        agent_result = agent_result_raw

    gold_result = None
    if _is_csv_like(gold_result_raw):
        gold_result = format_csv_as_table(gold_result_raw)
    else:
        gold_result = gold_result_raw

    # local verbose flag for run_diff_for_instance
    verbose = True

    # Try to load user query and external knowledge from jsonl if provided (strict)
    user_query = ""
    external_knowledge_file = None
    if jsonl_path:
        jsonl_path_obj = Path(jsonl_path)
        if not jsonl_path_obj.exists():
            raise FileNotFoundError(f"jsonl_path provided but not found: {jsonl_path}")
        with open(jsonl_path_obj, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                iid = obj.get('instance_id') or obj.get('id') or obj.get('tag')
                if iid and str(iid) == str(instance_id):
                    user_query = obj.get('question') or obj.get('query') or obj.get('question_text') or ''
                    external_knowledge_file = obj.get('external_knowledge') or obj.get('external_knowledge_file')
                    break

    # Load external knowledge if available
    external_knowledge = None
    if external_knowledge_file:
        external_knowledge = load_external_knowledge(instance_id, external_knowledge_file)
        if external_knowledge and verbose:
            if instance_id.lower().startswith("minidev"):
                preview = external_knowledge[:100] + "..." if len(external_knowledge) > 100 else external_knowledge
                print(f"[External Knowledge] Evidence text included: {preview}")
            else:
                print(f"[External Knowledge] Loaded from: {external_knowledge_file}")

    # Infer engine from instance_id and resolve db_path_or_cred accordingly
    engine = infer_engine(instance_id)
    db_path_or_cred = db_path  # Use provided db_path if available
    
    # If no explicit db_path provided, try to derive from jsonl using configured base_folder
    jsonl_to_use = jsonl_path or config.JSONL_DEFAULT
    data_base_folder = config.DATA_BASE_FOLDER
    if not db_path_or_cred and jsonl_to_use and Path(jsonl_to_use).exists():
        with open(jsonl_to_use, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                iid = obj.get('instance_id') or obj.get('id') or obj.get('tag')
                if iid and str(iid) == str(instance_id):
                    db_id = obj.get('db') or obj.get('database') or obj.get('db_id') or obj.get('database_id')
                    
                    # Resolve db_path_or_cred based on engine type
                    if engine == "sqlite" and db_id:
                        # Check minidev path first (if instance_id starts with "minidev")
                        if instance_id.lower().startswith("minidev"):
                            # Minidev databases are at: data/minidev/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite
                            project_root = Path(__file__).resolve().parent.parent
                            minidev_path = project_root / "data" / "minidev" / "MINIDEV" / "dev_databases" / db_id / f"{db_id}.sqlite"
                            if minidev_path.exists():
                                db_path_or_cred = str(minidev_path)
                            else:
                                print(f"⚠️  Warning: Minidev database not found at {minidev_path}")
                        # Fallback to standard spider2 path
                        elif data_base_folder:
                            example_folder = os.path.join(data_base_folder, str(instance_id))
                            if os.path.isdir(example_folder):
                                for file in os.listdir(example_folder):
                                    if file.endswith('.sqlite'):
                                        db_path_or_cred = os.path.join(example_folder, file)
                                        break
                    elif engine == "snowflake":
                        # Snowflake uses credential path (default provided by make_executor)
                        db_path_or_cred = "src/executors/snowflake_credential.json"
                    elif engine == "bq" or engine == "bigquery":
                        # BigQuery uses credential path (default provided by make_executor)
                        db_path_or_cred = "src/executors/bigquery_credential.json"
                    
                    # stop after first matching instance entry
                    break

    print(f"Running diff/repair for instance {instance_id} (engine={engine}) using outputs dir: {outputs_dir}")
    try:
        result = generate_memory_diff_first_turn(
            instance_id=instance_id,
            user_query=user_query,
            agent_cte_text="",  # not used in full-graph mode
            gold_sql_text=gold_sql,
            gold_result_csv_text=gold_result,
            agent_full_sql_text=agent_sql,
            agent_result_csv_text=agent_result,
            processed_trace_text=trace,
                engine=engine,
                db_path_or_cred=db_path_or_cred,
                db_name=db_id if (engine == "snowflake" or engine == "bq" or engine == "bigquery") else None,
                external_knowledge=external_knowledge,
            max_turns=max_turns,
            model=model,
            verbose=True,
            hint=hint,
        )
        print("\n=== FINAL OUTPUT ===")
        print(result)
        
        # Debug mode: write result to file and return early (skip tagging/indexing)
        if debug:
            debug_file = out / "debug_result.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== DIFF/REPAIR RESULT ===\n\n")
                f.write(result or "(no result)")
            print(f"\n[DEBUG MODE] Result written to: {debug_file}")
            print("[DEBUG MODE] Skipping rule extraction, tagging, and indexing.")
            return
        
    except Exception as e:
        # Check if this is a context window error that couldn't be recovered
        error_str = str(e)
        if "CONTEXT_WINDOW_EXCEEDED" in error_str or "context window" in error_str.lower():
            print(f"\n[ERROR] Context window exceeded for instance {instance_id}: {e}")
            # Log to failed_memories.csv
            idx_path = Path(config.FAILED_INDEX_PATH)
            header = ["instance_id", "outputs_dir", "reason", "clean_summary"]
            clean_summary = f"Context window exceeded during interactive repair. Error: {str(e)[:500]}"
            row = [str(instance_id), str(outputs_dir), "CONTEXT_WINDOW_EXCEEDED", clean_summary.replace('\n', ' ')]
            # ensure header
            if not idx_path.exists():
                with open(idx_path, 'w', encoding='utf-8', newline='') as wf:
                    writer = csv.writer(wf)
                    writer.writerow(header)
            with open(idx_path, 'a', encoding='utf-8', newline='') as af:
                writer = csv.writer(af)
                writer.writerow(row)
            print(f"[failed_index] appended instance {instance_id} (CONTEXT_WINDOW_EXCEEDED) to {idx_path}")
            # Raise the exception so batch script knows it failed
            raise Exception(f"CONTEXT_WINDOW_EXCEEDED: {str(e)}")
        # For other errors, re-raise so batch script can handle it
        raise

    # If MAX_TURNS reached -> record as failed and skip tagging/indexing
    try:
        if result and "MAX_TURNS_REACHED" in result:
            # extract CLEAN_SUMMARY if present
            clean_summary = ""
            try:
                m = re.search(r"CLEAN_SUMMARY\s*:\s*(.*)", result or "", flags=re.IGNORECASE)
                if m:
                    clean_summary = m.group(1).strip()
            except Exception:
                clean_summary = result.strip()[:2000]

            # append to failed CSV
            idx_path = Path(config.FAILED_INDEX_PATH)
            header = ["instance_id", "outputs_dir", "reason", "clean_summary"]
            row = [str(instance_id), str(outputs_dir), "MAX_TURNS_REACHED", clean_summary.replace('\n', ' ')]
            # ensure header
            if not idx_path.exists():
                with open(idx_path, 'w', encoding='utf-8', newline='') as wf:
                    writer = csv.writer(wf)
                    writer.writerow(header)
            with open(idx_path, 'a', encoding='utf-8', newline='') as af:
                writer = csv.writer(af)
                writer.writerow(row)
            print(f"[failed_index] appended instance {instance_id} to {idx_path}")
            return
    except Exception as e:
        # If failure to record failed entry, raise so caller sees it
        raise

    # Generate concise rules/checks/hints from the full diff/repair session output
    try:
        rules = generate_rules_from_diff(result or "", agent_sql, agent_result, model=model, verbose=False)
        print("\n=== GENERATED RULES (from diff) ===")
        print(rules or "(no rules generated)")
    except Exception as e:
        print(f"(rule generator failed: {e})")

    # Build and call the tagger to produce JSON for CLEAN_SUMMARY + memories
    try:
        # extract CLEAN_SUMMARY from result if present
        clean_summary = ""
        try:
            m = re.search(r"CLEAN_SUMMARY\s*:\s*(.*)", result or "", flags=re.IGNORECASE)
            if m:
                clean_summary = m.group(1).strip()
        except Exception:
            clean_summary = ""

        # parse question/generic memories from rules text
        q_mems = []
        g_mems = []
        if rules:
            try:
                # Support both QUESTION_MEMORIES (old) and DATABASE_MEMORIES (new) for backward compatibility
                # More robust regex: match DATABASE_MEMORIES: followed by content (greedy) until GENERIC_MEMORIES: or end
                mq = re.search(r"(?:DATABASE_MEMORIES|QUESTION_MEMORIES)\s*:?\s*(.*?)(?=\s*(?:GENERIC_MEMORIES|$))", rules, flags=re.S | re.IGNORECASE)
                if mq:
                    q_block = mq.group(1).strip()
                else:
                    q_block = ""
                # Match GENERIC_MEMORIES: followed by content until end or next major section
                mg = re.search(r"GENERIC_MEMORIES\s*:?\s*(.*?)(?=\s*(?:DATABASE_MEMORIES|QUESTION_MEMORIES|CLEAN_SUMMARY|$))", rules, flags=re.S | re.IGNORECASE)
                g_block = mg.group(1).strip() if mg else ""
                def _split_items(block: str) -> List[str]:
                    lines = []
                    for ln in block.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        # Skip if it's just the section header repeated
                        if ln.upper() in ["DATABASE_MEMORIES", "GENERIC_MEMORIES", "QUESTION_MEMORIES"]:
                            continue
                        # remove leading numbering like '1) ' or bullets like '- '
                        ln = re.sub(r"^[\d\)\-\•\*]\s*", "", ln)
                        if ln:  # Only add non-empty lines
                            lines.append(ln)
                    return lines
                q_mems = _split_items(q_block)
                g_mems = _split_items(g_block)
                # Always print extraction debug info to help diagnose issues
                print(f"[extraction] DATABASE_MEMORIES: extracted {len(q_mems)} items")
                print(f"[extraction] GENERIC_MEMORIES: extracted {len(g_mems)} items")
                if len(q_mems) == 0 and q_block:
                    print(f"[extraction] WARNING: q_block not empty but no items extracted. q_block preview: {q_block[:200]}")
                if len(g_mems) == 0 and g_block:
                    print(f"[extraction] WARNING: g_block not empty but no items extracted. g_block preview: {g_block[:200]}")
                # Also print what was matched to help debug
                if q_block and len(q_mems) == 0:
                    print(f"[extraction] DEBUG: q_block full content: {repr(q_block[:500])}")
                if g_block and len(g_mems) == 0:
                    print(f"[extraction] DEBUG: g_block full content: {repr(g_block[:500])}")
            except Exception:
                q_mems = []
                g_mems = []

        tagged = generate_tagged_memories_json(
            instance_id=str(instance_id),
            user_query=user_query,
            db_name=None,
            gold_sql=gold_sql,
            agent_sql=agent_sql,
            clean_summary=clean_summary,
            database_memories=q_mems,
            generic_memories=g_mems,
            evidence=( (result or "")[:2000] ),
            minimal_required_edits=None,
            model=model,
            verbose=True,
            multiturn=True,
        )
        if tagged is not None:
            print("\n=== TAGGED MEMORIES JSON ===")
            print(json.dumps(tagged, indent=2, ensure_ascii=False))
            try:
                # Build and append a clean index CSV for downstream search/indexing
                def _append_memories_index(tagged_obj: Dict[str, Any], out_dir: str, instance_id: str, verbose: bool = True):
                    idx_path = Path(config.MEMORY_INDEX_PATH)
                    header = ["mem_id", "instance_id", "scope", "sql_operations", "table", "column", "data_type", "nulls", "rule"]
                    rows = []

                    # If the LLM provided explicit index_rows, prefer them (trusted source)
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
                        # fallback: best-effort extraction from tagged_obj structure
                        def _normalize_ops(ops: Any) -> str:
                            if not ops:
                                return "all"
                            if isinstance(ops, list):
                                return ";".join([str(o).lower() for o in ops])
                            return str(ops).lower()

                        def _extract_table_col(data_objs: List[str]) -> Tuple[str, str]:
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
                            if re.search(r"\bnull\b", text, flags=re.I):
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
                                    # default: append minimal heuristic row
                                    rows.append([instance_id, scope_name, ops, table, column, data_type, nulls, text.replace('\n', ' ')])
                                except Exception:
                                    rows.append([instance_id, scope_name, "all", "all", "all", "unspecified", "No", str(mem)])

                    # write CSV (append)
                    try:
                        # determine if header exists; if missing, prepend it
                        write_header = False
                        start_idx = 0
                        if not idx_path.exists():
                            write_header = True
                            start_idx = 0
                        else:
                            # file exists: check first non-empty line
                            try:
                                first_line = None
                                with open(idx_path, 'r', encoding='utf-8') as rf:
                                    for ln in rf:
                                        if ln.strip():
                                            first_line = ln.strip()
                                            break
                                expected_header = ",".join(header)
                                if not first_line or (expected_header not in first_line and not first_line.startswith('mem_id')):
                                    # prepend header to existing file
                                    old = Path(idx_path).read_text(encoding='utf-8')
                                    with open(idx_path, 'w', encoding='utf-8', newline='') as wf:
                                        wf.write(expected_header + "\n")
                                        wf.write(old)
                                    # after prepending, start_idx equals existing non-empty lines count
                                    existing = [ln for ln in old.splitlines() if ln.strip()]
                                    start_idx = len(existing)
                                else:
                                    # compute start_idx based on existing rows (subtract header)
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

                _append_memories_index(tagged, outputs_dir, str(instance_id), verbose=True)
                # Also store CLEAN_SUMMARY as a question-scoped rule with no tags (preserve verbatim)
                try:
                    if clean_summary:
                        idx_path = Path(config.MEMORY_INDEX_PATH)
                        header = ["mem_id", "instance_id", "scope", "sql_operations", "table", "column", "data_type", "nulls", "rule"]
                        # ensure header exists
                        if not idx_path.exists():
                            with open(idx_path, 'w', encoding='utf-8', newline='') as wf:
                                writer = csv.writer(wf)
                                writer.writerow(header)
                        # count existing data rows
                        with open(idx_path, 'r', encoding='utf-8') as rf:
                            existing = [ln for ln in rf.read().splitlines() if ln.strip()]
                        data_rows = max(0, len(existing) - 1)
                        mem_id = data_rows
                        # Mark CLEAN_SUMMARY rows as non-searchable using reserved token 'NA'
                        # Ensure we emit exactly 9 columns matching the header: mem_id, instance_id, scope, sql_operations, table, column, data_type, nulls, rule
                        row = [mem_id, str(instance_id), "question", "NA", "NA", "NA", "NA", "NA", clean_summary.replace('\n', ' ')]
                        with open(idx_path, 'a', encoding='utf-8', newline='') as af:
                            writer = csv.writer(af)
                            writer.writerow(row)
                        if verbose:
                            print(f"[index] appended CLEAN_SUMMARY as mem_id={mem_id} to {idx_path}")
                except Exception as e:
                    if verbose:
                        print(f"(failed to append CLEAN_SUMMARY to index: {e})")
            except Exception as e:
                print(f"(indexing failed: {e})")
    except Exception as e:
        print(f"(tagger failed: {e})")



