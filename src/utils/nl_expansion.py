"""NL() UDF expansion — translates NL("description", "schema") calls into real SQL subqueries."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


# --------------- Schema Context ---------------

_schema_cache: Dict[str, str] = {}


def get_schema_context(executor: Executor) -> str:
    """Retrieve database schema as text for LLM context.

    Caches by executor db_path to avoid repeated queries within a session.
    """
    cache_key = getattr(executor, 'db_path', None) or id(executor)
    cache_key = str(cache_key)
    if cache_key in _schema_cache:
        return _schema_cache[cache_key]

    lines = []
    try:
        _, tables = executor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        for (table_name,) in tables:
            lines.append(f"TABLE: {table_name}")
            try:
                _, cols = executor.execute(f"PRAGMA table_info({table_name})")
                for col in cols:
                    # col: (cid, name, type, notnull, default, pk)
                    col_name = col[1]
                    col_type = col[2] or "TEXT"
                    lines.append(f"  {col_name} {col_type}")
            except Exception:
                lines.append("  (schema unavailable)")
            lines.append("")
    except Exception as e:
        lines.append(f"(schema retrieval failed: {e})")

    schema_text = "\n".join(lines)
    _schema_cache[cache_key] = schema_text
    return schema_text


def clear_schema_cache():
    """Clear the cached schema context."""
    _schema_cache.clear()


# --------------- NL-to-SQL Translation ---------------

NL_TRANSLATE_SYSTEM_PROMPT = """\
You are a SQL generation assistant. Given a natural language description and a database schema, \
generate a single SELECT query that returns the described data.

RULES:
- Return ONLY a single SELECT statement. No explanations, no markdown, no extra text.
- The result columns MUST match this output schema exactly: {output_schema}
- Use the column names and types specified in the output schema.
- Use only tables and columns that exist in the database schema below.
- Do NOT use NL() or any custom functions in your generated SQL.
- Write standard SQLite-compatible SQL.

DATABASE SCHEMA:
{schema_context}
"""


def translate_nl_to_sql(
    description: str,
    output_schema: str,
    schema_context: str,
    model: str,
) -> str:
    """Translate a natural language description into a SQL SELECT statement."""
    from src.agents.sql_agent_runner import llm_completion

    system_msg = NL_TRANSLATE_SYSTEM_PROMPT.format(
        output_schema=output_schema,
        schema_context=schema_context,
    )
    user_msg = f"Generate SQL for: {description}\nRequired output columns: {output_schema}"

    resp = llm_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = (resp["choices"][0]["message"].get("content") or "").strip()

    # Strip markdown fences if present
    content = re.sub(r'^```\s*sql\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'^```\s*', '', content)
    content = re.sub(r'```\s*$', '', content)
    content = content.strip()

    # Strip trailing semicolons
    content = content.rstrip(';').strip()

    return content


# --------------- Main Expansion ---------------

def expand_nl_calls(
    sql: str,
    executor: Executor,
    model: str,
    schema_context: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[str, Optional[str]]:
    """Expand all NL() calls in a SQL string into real subqueries.

    Returns (expanded_sql, schema_context) where schema_context is returned
    so it can be cached by the caller for subsequent calls.
    """
    calls = find_nl_calls(sql)
    if not calls:
        return sql, schema_context

    if schema_context is None:
        schema_context = get_schema_context(executor)

    # Process in reverse order to preserve string positions
    expanded = sql
    for call in reversed(calls):
        try:
            generated_sql = translate_nl_to_sql(
                description=call.description,
                output_schema=call.output_schema,
                schema_context=schema_context,
                model=model,
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

    return expanded, schema_context


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
