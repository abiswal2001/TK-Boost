"""NL() UDF expansion — translates NL("description", "schema") calls into real SQL subqueries."""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.executors.base import Executor


@dataclass
class NLCall:
    """A parsed NL() invocation found in a SQL string."""
    full_match: str
    start: int
    end: int
    description: str
    output_schema: str


def find_nl_calls(sql: str) -> List[NLCall]:
    """Find all NL("...", "...") calls in a SQL string.

    Uses a state machine to avoid matching inside string literals.
    Returns calls sorted by start position.
    """
    calls: List[NLCall] = []
    n = len(sql)
    i = 0

    while i < n:
        # Skip string literals
        if sql[i] in ("'", '"'):
            quote = sql[i]
            i += 1
            while i < n:
                if sql[i] == quote:
                    i += 1
                    break
                if sql[i] == '\\':
                    i += 1  # skip escaped char
                i += 1
            continue

        # Skip single-line comments
        if sql[i:i+2] == '--':
            while i < n and sql[i] != '\n':
                i += 1
            continue

        # Skip block comments
        if sql[i:i+2] == '/*':
            i += 2
            while i < n - 1 and sql[i:i+2] != '*/':
                i += 1
            i += 2
            continue

        # Look for NL( — must be a word boundary before NL
        if sql[i:i+2].upper() == 'NL' and (i == 0 or not sql[i-1].isalnum() and sql[i-1] != '_'):
            # Check for opening paren (possibly with whitespace)
            j = i + 2
            while j < n and sql[j] == ' ':
                j += 1
            if j < n and sql[j] == '(':
                call = _parse_nl_call(sql, i, j)
                if call:
                    calls.append(call)
                    i = call.end
                    continue

        i += 1

    return calls


def _parse_nl_call(sql: str, start: int, paren_pos: int) -> Optional[NLCall]:
    """Parse NL(...) starting at `start` with opening paren at `paren_pos`.

    Expects exactly two quoted-string arguments separated by a comma.
    """
    n = len(sql)
    j = paren_pos + 1  # after '('

    # Parse first argument (quoted string)
    arg1, j = _parse_quoted_arg(sql, j)
    if arg1 is None:
        return None

    # Skip whitespace and comma
    while j < n and sql[j] in (' ', '\t', '\n', '\r'):
        j += 1
    if j >= n or sql[j] != ',':
        return None
    j += 1  # skip comma

    # Parse second argument (quoted string)
    arg2, j = _parse_quoted_arg(sql, j)
    if arg2 is None:
        return None

    # Skip whitespace then expect closing paren
    while j < n and sql[j] in (' ', '\t', '\n', '\r'):
        j += 1
    if j >= n or sql[j] != ')':
        return None
    j += 1  # skip ')'

    return NLCall(
        full_match=sql[start:j],
        start=start,
        end=j,
        description=arg1,
        output_schema=arg2,
    )


def _parse_quoted_arg(sql: str, pos: int) -> Tuple[Optional[str], int]:
    """Parse a quoted string argument starting at pos (skipping leading whitespace).

    Returns (parsed_string, new_position) or (None, pos) on failure.
    """
    n = len(sql)
    j = pos

    # Skip whitespace
    while j < n and sql[j] in (' ', '\t', '\n', '\r'):
        j += 1

    if j >= n or sql[j] not in ('"', "'"):
        return None, pos

    quote = sql[j]
    j += 1
    chars = []
    while j < n:
        if sql[j] == '\\':
            j += 1
            if j < n:
                chars.append(sql[j])
                j += 1
            continue
        if sql[j] == quote:
            # Check for escaped quote (doubled)
            if j + 1 < n and sql[j+1] == quote:
                chars.append(quote)
                j += 2
                continue
            j += 1  # skip closing quote
            return ''.join(chars), j
        chars.append(sql[j])
        j += 1

    return None, pos  # unterminated string


# --------------- Sub-Agent Translation ---------------

def run_sub_agent(
    description: str,
    output_schema: str,
    executor: Executor,
    model: str,
    verbose: bool = False,
    trace_dir: Optional[str] = None,
    main_turn: Optional[int] = None,
    nl_call_index: int = 0,
) -> str:
    """Run a full ReAct agent to translate a natural language description into SQL."""
    from src.agents.sql_agent_runner import Instance, run_agent, generate_processed_trace
    from src.agents.prompts import SUB_AGENT_PROMPT
    import json

    question = (
        f"{description}\n"
        f"The output must have exactly these columns: {output_schema}"
    )

    engine = "sqlite"  # NL() UDF currently supports SQLite
    db_path_or_cred = getattr(executor, 'db_path', None)

    inst = Instance(
        instance_id="nl_sub_agent",
        db="",
        question=question,
    )

    if verbose:
        print(f"\n{'─'*40}")
        print(f"[NL() SUB-AGENT START]: {description}")
        print(f"{'─'*40}")

    final_sql, _headers, _rows, _messages, _exec = run_agent(
        inst=inst,
        engine=engine,
        db_path_or_cred=db_path_or_cred,
        model=model,
        predicted_cte_hint=None,
        predicted_schema_hint=None,
        max_turns=10,
        verbose=verbose,
        system_prompt=SUB_AGENT_PROMPT,
    )

    # Save sub-agent trace into a shared sub_agents/ directory
    if trace_dir:
        from pathlib import Path
        sub_dir = Path(trace_dir) / "sub_agents"
        sub_dir.mkdir(parents=True, exist_ok=True)
        turn_label = f"turn{main_turn}" if main_turn is not None else "final"
        prefix = f"{turn_label}_nl{nl_call_index}"
        (sub_dir / f"{prefix}_messages.json").write_text(
            json.dumps(_messages, indent=2), encoding="utf-8"
        )
        (sub_dir / f"{prefix}_trace.txt").write_text(
            generate_processed_trace(_messages), encoding="utf-8"
        )
        (sub_dir / f"{prefix}_query.sql").write_text(
            final_sql or "", encoding="utf-8"
        )
        if verbose:
            print(f"📁 Sub-agent trace saved to: {sub_dir}/{prefix}_*")

    if verbose:
        print(f"\n{'─'*40}")
        print(f"[NL() SUB-AGENT DONE]: {final_sql[:200] if final_sql else '(no SQL)'}")
        print(f"{'─'*40}")

    # Strip trailing semicolons
    if final_sql:
        final_sql = final_sql.rstrip(';').strip()

    return final_sql or ""


# --------------- Main Expansion ---------------

def expand_nl_calls(
    sql: str,
    executor: Executor,
    model: str,
    verbose: bool = False,
    trace_dir: Optional[str] = None,
    main_turn: Optional[int] = None,
) -> str:
    """Expand all NL() calls in a SQL string into real subqueries.

    Each NL() call is translated by running a full sub-agent (ReAct loop).
    Returns the expanded SQL string.
    """
    calls = find_nl_calls(sql)
    if not calls:
        return sql

    # Process in reverse order to preserve string positions
    expanded = sql
    for call_idx, call in enumerate(reversed(calls)):
        try:
            generated_sql = run_sub_agent(
                description=call.description,
                output_schema=call.output_schema,
                executor=executor,
                model=model,
                verbose=verbose,
                trace_dir=trace_dir,
                main_turn=main_turn,
                nl_call_index=len(calls) - 1 - call_idx,
            )

            # Parse output_schema to get column names for the wrapper
            col_names = _parse_schema_columns(call.output_schema)
            if col_names:
                wrapper = f"(SELECT {', '.join(col_names)} FROM ({generated_sql}) AS _nl_inner)"
            else:
                wrapper = f"({generated_sql})"

            if verbose:
                print(f"\n[NL() EXPANSION]: NL(\"{call.description}\", \"{call.output_schema}\")")
                print(f"  -> {generated_sql[:200]}{'...' if len(generated_sql) > 200 else ''}")

            expanded = expanded[:call.start] + wrapper + expanded[call.end:]

        except Exception as e:
            if verbose:
                print(f"\n[NL() EXPANSION FAILED]: {e}")
            # Leave original NL() call in place — executor will error,
            # agent gets SQL_ERROR feedback and can retry
            pass

    return expanded


def _parse_schema_columns(output_schema: str) -> List[str]:
    """Extract column names from an output schema string like 'col1 TYPE, col2 TYPE'."""
    cols = []
    for part in output_schema.split(','):
        part = part.strip()
        if part:
            # First token is the column name
            name = part.split()[0].strip()
            if name:
                cols.append(name)
    return cols
