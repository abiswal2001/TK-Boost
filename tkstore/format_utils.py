import re
import csv
from typing import List, Dict, Any, Optional


def format_csv_as_table(csv_text: Optional[str]) -> Optional[str]:
    """Convert CSV text (header+rows) into a Markdown-style table string.

    - If input looks like an SQL error message or is empty, return it unchanged.
    - On parse errors, fall back to returning the original text.
    """
    if not csv_text:
        return None
    # Preserve error strings or harness messages as-is
    if str(csv_text).startswith("SQL_ERROR:") or str(csv_text).startswith("SQL_REJECTED_STATIC_OUTPUT:"):
        return csv_text
    try:
        lines = [l for l in str(csv_text).splitlines() if l.strip() != ""]
        if not lines:
            return csv_text
        reader = csv.reader(lines)
        rows = list(reader)
        if not rows:
            return csv_text

        # Normalize rows to the same column length and detect whether the
        # first row is a header or actually contains numeric data (common
        # when CSVs lack headers). If the first row looks numeric-only,
        # treat it as data and synthesize generic headers `col1`, `col2`, ...
        num_cols = max(len(r) for r in rows)
        # pad rows to same length
        norm_rows = [r + [""] * (num_cols - len(r)) for r in rows]

        header_candidate = [c.strip() for c in norm_rows[0]]

        def _is_numeric_cell(s: str) -> bool:
            if s is None:
                return False
            s = s.strip()
            if s == "":
                return False
            return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)$", s))

        # If every header cell looks numeric, we assume there's no header
        # row and synthesize generic column names. Otherwise use the first
        # row as header and the remaining rows as body.
        if all(_is_numeric_cell(c) for c in header_candidate):
            header = [f"   {i}" for i in range(num_cols)]
            body = norm_rows
        else:
            header = header_candidate
            body = norm_rows[1:]

        # Drop a common "index" row that some systems emit as the first
        # data row (e.g. `0,1,2,3,...`). If the first body row is exactly
        # the sequence 0..N-1 or 1..N, remove it as it's a misleading header.
        if body:
            first_row = [c.strip() for c in body[0]]
            try:
                if all(re.fullmatch(r"[+-]?\d+", c) for c in first_row):
                    nums = [int(c) for c in first_row]
                    if nums == list(range(len(first_row))) or nums == list(range(1, len(first_row) + 1)):
                        body = body[1:]
            except Exception:
                # on any unexpected parse issue, leave body unchanged
                pass

        def esc(cell: Any) -> str:
            if cell is None:
                return ""
            return str(cell).replace("|", "\\|")

        # Compute column widths for aligned, stable formatting
        columns = [header] + body
        col_widths = []
        for col_idx in range(len(header)):
            max_w = 0
            for row in columns:
                cell = esc(row[col_idx]) if col_idx < len(row) else ""
                max_w = max(max_w, len(cell))
            col_widths.append(max_w)

        def _pad(cell: Any, width: int) -> str:
            s = esc(cell) if cell is not None else ""
            return s + (" " * (width - len(s)))

        header_line = "| " + " | ".join([_pad(h, col_widths[i]) for i, h in enumerate(header)]) + " |"
        sep_line = "| " + " | ".join(["-" * col_widths[i] for i in range(len(header))]) + " |"
        body_lines = ["| " + " | ".join([_pad(row[i] if i < len(row) else "", col_widths[i]) for i in range(len(header))]) + " |" for row in body]
        table = "\n".join([header_line, sep_line] + body_lines)
        return table
    except Exception:
        return csv_text


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


