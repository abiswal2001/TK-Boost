import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import datetime


def format_table(headers: Optional[List[str]], rows: List[Tuple], limit: int = 200) -> str:
    if not rows:
        return "<empty>"
    headers = headers or [f"col{i+1}" for i in range(len(rows[0]))]
    sep = "-+-".join(["-" * len(h) for h in headers])
    lines = [" | ".join(headers), sep]
    for r in rows[:limit]:
        lines.append(" | ".join([str(x) for x in r]))
    return "\n".join(lines)


def detect_sql_blocks(content: str) -> List[str]:
    blocks = re.findall(r"<sql>(.*?)</sql>", content, flags=re.DOTALL | re.IGNORECASE)
    fences = re.findall(r"```\s*sql\s*([\s\S]*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if not blocks and not fences:
        generic = re.findall(r"```\s*([\s\S]*?)```", content, flags=re.DOTALL)
        for blk in generic:
            b = blk.strip()
            low = b.lower()
            if any(k in low for k in ["select ", "pragma ", "with ", "explain "]):
                fences.append(b)
    return blocks + fences


def detect_solution(content: str) -> Optional[str]:
    m = re.search(r"<solution>([\s\S]*?)</solution>", content, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_csv(headers: Optional[List[str]], rows: List[Tuple], dest: Path) -> None:
    with open(dest, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        writer = _csv.writer(f)
        if headers:
            writer.writerow(headers)
        for r in rows:
            writer.writerow(list(r))


def make_json_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to serializable ones.
    
    Handles Decimal, datetime objects, and bytes commonly returned by database connectors.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    return obj


def load_external_knowledge(instance_id: str, external_knowledge_file: Optional[str]) -> Optional[str]:
    """Load external knowledge file if specified for an instance.
    
    Args:
        instance_id: The instance identifier
        external_knowledge_file: Filename of the external knowledge (e.g., "context.md")
        
    Returns:
        Content of the external knowledge file, or None if not available
    """
    if not external_knowledge_file:
        return None
    
    ext_file = external_knowledge_file.strip()
    if not ext_file:
        return None
    
    # External knowledge files are in data/spider2/{instance_id}/{filename}
    ext_path = Path("data/spider2") / instance_id / ext_file
    if ext_path.exists():
        try:
            return ext_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️  Warning: Could not read external knowledge file {ext_path}: {e}")
            return None
    return None


def load_predicted_cte_briefs(csv_path: Optional[str]) -> Dict[str, str]:
    if not csv_path:
        return {}
    out: Dict[str, str] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = (row.get("instance_id") or "").strip()
                briefs = (row.get("predicted_cte_briefs") or row.get("raw_output") or "").strip()
                if iid and briefs:
                    out[iid] = briefs
    except Exception as e:
        print(f"⚠️  Could not load predicted CTE briefs '{csv_path}': {e}")
    return out


def load_predicted_tables_columns(csv_path: Optional[str]) -> Dict[str, str]:
    import re
    if not csv_path:
        return {}
    out: Dict[str, str] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = (row.get("instance_id") or "").strip()
                text = (row.get("predicted_tables_and_columns") or row.get("raw_output") or "").strip()
                
                # For BigQuery instances, if predicted_tables_and_columns is empty, 
                # construct from predicted_tables_fq and predicted_columns_fq
                if iid and not text and (iid.lower().startswith('bq') or iid.lower().startswith('ga')):
                    tables_fq = (row.get("predicted_tables_fq") or "").strip()
                    columns_fq = (row.get("predicted_columns_fq") or "").strip()
                    
                    if tables_fq or columns_fq:
                        # Fix BigQuery table names to use bigquery-public-data prefix
                        # Replace patterns like: dataset.schema.table -> bigquery-public-data.dataset.table
                        # But preserve if already has bigquery-public-data
                        if tables_fq:
                            # Split by comma to handle multiple tables
                            fixed_tables = []
                            for table in tables_fq.split(','):
                                table = table.strip()
                                if table and not table.startswith('bigquery-public-data.'):
                                    # Remove any incorrect project prefix (e.g., ga360.ga360.table -> ga360.table)
                                    parts = table.split('.')
                                    if len(parts) >= 2:
                                        # Take last 2 parts (dataset.table or dataset.table*)
                                        table = '.'.join(parts[-2:])
                                        # Now prepend bigquery-public-data
                                        table = f'bigquery-public-data.{table}'
                                if table:
                                    fixed_tables.append(table)
                            tables_fq = ', '.join(fixed_tables)
                        
                        if columns_fq:
                            # Fix column references similarly
                            fixed_cols = []
                            for col in columns_fq.split(';'):
                                col = col.strip()
                                if col and not col.startswith('bigquery-public-data.'):
                                    # Split table.column
                                    if '.' in col:
                                        parts = col.split('.')
                                        if len(parts) >= 3:
                                            # Take last 3 parts (dataset.table.column)
                                            col_part = parts[-1]
                                            table_part = '.'.join(parts[-3:-1])
                                            # Remove duplicate dataset prefix
                                            table_parts = table_part.split('.')
                                            if len(table_parts) >= 2:
                                                table_part = '.'.join(table_parts[-2:])
                                            col = f'bigquery-public-data.{table_part}.{col_part}'
                                if col:
                                    fixed_cols.append(col)
                            columns_fq = '; '.join(fixed_cols)
                        
                        # Construct the text
                        parts = []
                        if tables_fq:
                            parts.append(f"Tables: {tables_fq}")
                        if columns_fq:
                            parts.append(f"Columns: {columns_fq}")
                        text = "\n".join(parts)
                
                if iid and text:
                    out[iid] = text
    except Exception as e:
        print(f"⚠️  Could not load predicted tables/columns '{csv_path}': {e}")
    return out


def infer_engine(instance_id: str) -> str:
    low = (instance_id or "").lower()
    if low.startswith("local") or low.startswith("minidev"):
        return "sqlite"
    if low.startswith("bq") or low.startswith("ga"):
        return "bq"
    if low.startswith("sf"):
        return "snowflake"
    return "sqlite"


def parse_ctes_from_sql(sql_text: str) -> Tuple[List[Dict[str, str]], str]:
    """Best-effort parse of CTEs and remainder SELECT.

    Returns (ctes, remainder_sql) where ctes is a list of {name, body}.
    """
    text = sql_text or ""
    i = 0
    n = len(text)
    ctes: List[Dict[str, str]] = []
    remainder = text
    # Seek 'with'
    m = re.search(r"\bwith\b", text, flags=re.IGNORECASE)
    if not m:
        return ctes, text
    i = m.start()
    # From 'with' to the end, parse name AS ( ... ) blocks separated by commas until we reach a SELECT
    j = i + 4
    while j < n:
        # skip whitespace, commas, and comments
        while j < n:
            if text[j] in " \t\n,":
                j += 1
            elif j + 1 < n and text[j:j+2] == '--':
                # Skip comment until end of line
                while j < n and text[j] != '\n':
                    j += 1
                if j < n:
                    j += 1  # skip the newline
            else:
                break
        # read name
        name_start = j
        while j < n and re.match(r"[A-Za-z0-9_]", text[j] or " "):
            j += 1
        name = text[name_start:j].strip()
        if not name:
            break
        # skip whitespace
        while j < n and text[j].isspace():
            j += 1
        # expect 'as'
        if not re.match(r"(?i)as", text[j:j+2] or ""):
            break
        j += 2
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != '(':
            break
        # capture balanced parentheses for body
        depth = 0
        body_start = j + 1
        while j < n:
            if text[j] == '(':
                depth += 1
            elif text[j] == ')':
                depth -= 1
                if depth == 0:
                    # body ends before current ')'
                    body = text[body_start:j].strip()
                    ctes.append({"name": name, "body": body})
                    j += 1
                    break
            j += 1
        else:
            break
        # peek next token; if SELECT likely remainder
        lookahead = text[j:j+20].lower()
        if re.search(r"\bselect\b", lookahead):
            remainder = text[j:].strip()
            break
        # else loop for next CTE name
    if not ctes:
        return [], text
    # If remainder wasn't set inside loop, try to find SELECT after last position
    if remainder == text:
        m2 = re.search(r"\bselect\b", text[j:], flags=re.IGNORECASE)
        if m2:
            remainder = text[j+m2.start():].strip()
        else:
            remainder = ""
    return ctes, remainder


def rebuild_sql_from_ctes(ctes: List[Dict[str, str]], remainder_sql: str) -> str:
    if not ctes:
        return remainder_sql or ""
    parts = []
    for c in ctes:
        cname = c.get('name') or ''
        cbody = c.get('body') or ''
        parts.append(f"{cname} AS (\n{cbody}\n)")
    with_block = "WITH " + ",\n".join(parts)
    if remainder_sql and remainder_sql.strip():
        return with_block + "\n" + remainder_sql.strip()
    return with_block


def extract_goal_from_cte_body(cte_body: str, cte_name: str) -> str:
    # Heuristic: read comment lines containing 'Goal' or the first sentence
    for line in (cte_body or '').splitlines():
        low = line.strip().lower()
        if low.startswith('-- goal:') or 'goal' in low:
            return line.strip().lstrip('-').strip()
    return f"CTE {cte_name} validation"


