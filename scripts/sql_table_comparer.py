#!/usr/bin/env python3
"""
Improved SQL Table Comparer
===========================

Improved version that focuses on actual database columns rather than aliases.
"""

import os
import re
import json
import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import litellm


# ----------------- Configuration -----------------
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE", "https://east-docetl.openai.azure.com/")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")

# Set up environment
if AZURE_API_KEY:
    os.environ["AZURE_API_KEY"] = AZURE_API_KEY
if AZURE_API_BASE:
    os.environ["AZURE_API_BASE"] = AZURE_API_BASE
if AZURE_API_VERSION:
    os.environ["AZURE_API_VERSION"] = AZURE_API_VERSION


def read_sql_from_file(file_path: str) -> str:
    """Read SQL content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f"Failed to read SQL from {file_path}: {e}")


def load_hardcoded_schemas() -> Dict[str, Dict[str, List[Dict]]]:
    """Load hardcoded schemas from JSON file."""
    schema_file = "hardcoded_bigquery_schemas.json"
    if os.path.exists(schema_file):
        try:
            with open(schema_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load hardcoded schemas: {e}")
    return {}

def get_database_schema_for_instance(instance_id: str) -> Dict[str, List[Dict]]:
    """Get database schema for the given instance_id."""
    # First check for hardcoded schemas
    hardcoded_schemas = load_hardcoded_schemas()
    if instance_id in hardcoded_schemas:
        return hardcoded_schemas[instance_id]
    
    # For BigQuery instances, look for DDL CSV files
    tasks_dir = f"data/spider2/{instance_id}"
    
    if os.path.exists(tasks_dir):
        return get_bigquery_schema_from_ddl(tasks_dir)
    
    # Fallback to SQLite for local instances
    db_name = None
    try:
        with open('questions_bq/spider2-lite.jsonl', 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    if data.get('instance_id') == instance_id:
                        db_name = data.get('db')
                        break
    except Exception as e:
        print(f"⚠️  Could not read questions file: {e}")
    
    if not db_name:
        raise FileNotFoundError(f"Could not find database name for instance {instance_id}")
    
    # Look for the database file using the correct path
    db_path = f"/Users/cusgadmin/Documents/StructuredMemory/structuredagentmemory/data/{instance_id}/{db_name}.sqlite"
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        schema_info = {}
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            schema_info[table_name] = []
            for col in columns:
                col_id, name, col_type, not_null, default_val, pk = col
                schema_info[table_name].append({
                    'name': name,
                    'type': col_type,
                    'not_null': bool(not_null),
                    'default': default_val,
                    'primary_key': bool(pk)
                })
        
        conn.close()
        return schema_info
        
    except Exception as e:
        print(f"❌ Error getting schema for {instance_id}: {e}")
        return {}


def get_bigquery_schema_from_ddl(tasks_dir: str) -> Dict[str, List[Dict]]:
    """Extract schema from BigQuery DDL CSV files, consolidating similar tables."""
    import csv
    import re
    
    schema_info = {}
    
    # Find all DDL.csv files
    ddl_files = []
    for root, dirs, files in os.walk(tasks_dir):
        for file in files:
            if file.lower() == 'ddl.csv':
                ddl_files.append(os.path.join(root, file))
    
    if not ddl_files:
        print(f"⚠️  No DDL.csv files found in {tasks_dir}")
        return {}
    
    print(f"📋 Found {len(ddl_files)} DDL files")
    
    # Process each DDL file
    for ddl_file in ddl_files:
        try:
            with open(ddl_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    table_name = row['table_name']
                    ddl = row['ddl']
                    
                    # Extract columns from DDL
                    columns = extract_columns_from_ddl(ddl)
                    
                    if columns:
                        schema_info[table_name] = columns
                        
        except Exception as e:
            print(f"⚠️  Error reading DDL file {ddl_file}: {e}")
            continue
    
    # Consolidate similar tables (e.g., date-partitioned tables)
    consolidated_schema = consolidate_similar_tables(schema_info)
    
    return consolidated_schema


def extract_columns_from_ddl(ddl: str) -> List[Dict]:
    """Extract column information from a BigQuery DDL string."""
    columns = []
    
    # Simple regex to extract column definitions
    # This is a basic implementation - might need refinement for complex DDLs
    column_pattern = r'(\w+)\s+([A-Z0-9_]+(?:\([^)]*\))?)'
    
    # Find the CREATE TABLE part and extract columns
    create_match = re.search(r'CREATE TABLE[^(]*\(([^)]+)\)', ddl, re.IGNORECASE | re.DOTALL)
    if not create_match:
        return columns
    
    table_content = create_match.group(1)
    
    # Split by comma but be careful with nested structures
    parts = []
    current_part = ""
    paren_count = 0
    
    for char in table_content:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == ',' and paren_count == 0:
            parts.append(current_part.strip())
            current_part = ""
            continue
        current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    # Extract column info from each part
    for part in parts:
        part = part.strip()
        if not part or part.startswith('PRIMARY KEY') or part.startswith('PARTITION BY'):
            continue
            
        # Extract column name and type
        match = re.match(r'^(\w+)\s+([A-Z0-9_]+(?:\([^)]*\))?)', part)
        if match:
            col_name = match.group(1)
            col_type = match.group(2)
            
            columns.append({
                'name': col_name,
                'type': col_type,
                'not_null': False,  # Simplified for BigQuery
                'default': None,
                'primary_key': False
            })
    
    return columns


def consolidate_similar_tables(schema_info: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Consolidate similar tables (e.g., date-partitioned tables) into representative schemas."""
    consolidated = {}
    
    # Group tables by base name (remove date suffixes, etc.)
    table_groups = {}
    
    for table_name in schema_info.keys():
        # Extract base table name by removing common suffixes
        base_name = table_name
        
        # Remove date patterns (e.g., _20170127, _20160801)
        base_name = re.sub(r'_\d{8}$', '', base_name)
        base_name = re.sub(r'_\d{6}$', '', base_name)
        base_name = re.sub(r'_\d{4}$', '', base_name)
        
        # Remove other common suffixes
        base_name = re.sub(r'_\d+$', '', base_name)  # Remove numeric suffixes
        
        if base_name not in table_groups:
            table_groups[base_name] = []
        table_groups[base_name].append(table_name)
    
    # For each group, use the first table's schema and add _* suffix
    for base_name, table_list in table_groups.items():
        if len(table_list) > 1:
            # Multiple similar tables - consolidate
            representative_table = table_list[0]
            consolidated_name = f"{base_name}_*"
            consolidated[consolidated_name] = schema_info[representative_table]
            print(f"📊 Consolidated {len(table_list)} tables into {consolidated_name}")
        else:
            # Single table - keep as is
            consolidated[base_name] = schema_info[table_list[0]]
    
    return consolidated


def extract_tables_with_llm(sql: str, model: str = "azure/gpt-4o", schema_info: Dict = None) -> List[str]:
    """Use LLM to extract base table/view identifiers from SQL."""
    schema_context = ""
    if schema_info:
        schema_context = f"\n\nDATABASE SCHEMA:\n{json.dumps(schema_info, indent=2)}\n\n"
    
    system_prompt = (
        "You are a precise SQL structure parser. Given one SQL query, identify ONLY SOURCE DATABASE TABLES/VIEWS "
        "that are referenced in FROM or JOIN clauses. "
        "\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. First, identify ALL table names defined in WITH clauses (these are CTEs)\n"
        "2. Then, identify ALL table names referenced in FROM/JOIN clauses\n"
        "3. EXCLUDE any table name that appears in step 1 (CTEs)\n"
        "4. ONLY INCLUDE TABLE NAMES THAT APPEAR IN THE DATABASE SCHEMA BELOW AND ARE NOT CTEs\n"
        "\n"
        f"{schema_context}"
        "Return ONLY JSON with the schema:\n"
        "{\"tables\": [\"...\", \"...\"]}"
    )

    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"SQL:\n{sql}"},
            ],
            temperature=1.0 if "o4-mini" in model or "o3-mini" in model or "o3" in model else 0.0,
        )
        content = resp.choices[0].message.content.strip()
        # Strip markdown fences if any
        if content.startswith("```json"):
            content = content[len("```json"):]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        parsed = json.loads(content)
        tables = parsed.get("tables", [])
        # Normalize to strings and strip whitespace
        out: List[str] = []
        seen = set()
        for t in tables:
            if not isinstance(t, str):
                continue
            tt = t.strip()
            if tt and tt not in seen:
                out.append(tt)
                seen.add(tt)
        
        # Manual filtering: only include tables that exist in the schema
        if schema_info:
            schema_tables = set(schema_info.keys())
            filtered_tables = [table for table in out if table in schema_tables]
            print(f"🔍 LLM extracted: {out}")
            print(f"🔍 Schema tables: {list(schema_tables)}")
            print(f"🔍 Filtered result: {filtered_tables}")
            return filtered_tables
        else:
            print(f"🔍 No schema info provided, returning LLM result: {out}")
            return out
    except Exception as e:
        print(f"⚠️  LLM extraction failed: {e}")
        # Fallback regex
        try:
            sql_no_cte = re.sub(r"^\s*with\s+.*?\)\s*select", " select", sql, flags=re.IGNORECASE | re.DOTALL)
        except Exception:
            sql_no_cte = sql
        pattern = r"(?:from|join)\s+([`\"]?[\w\.\-]+[`\"]?)"
        candidates = re.findall(pattern, sql_no_cte, flags=re.IGNORECASE)
        cleaned = []
        seen = set()
        for c in candidates:
            base = c.strip().split()[0]
            if base and base not in seen:
                cleaned.append(base)
                seen.add(base)
        return cleaned


def extract_database_columns_with_llm(sql: str, model: str = "azure/gpt-4o", schema_info: Dict = None) -> List[str]:
    """Extract ACTUAL database columns and resolve them to real base tables using the provided schema.

    Strategy (robust):
    1) If schema_info is available, first try an LLM prompt that resolves alias/CTE usage to real base tables.
       Return fully-qualified names: RealTable.column. Filter strictly to columns present in schema.
    2) If that fails, fall back to a simpler column listing prompt, then uniquely qualify by schema (strip aliases, map
       only when a column exists in exactly one table). Drop ambiguous matches.
    3) If no schema is provided, return the simple list (deduped) as best-effort.
    """
    # Attempt schema-aware resolution via LLM when schema is present
    if schema_info:
        try:
            schema_context = json.dumps(schema_info, indent=2)
            system_prompt_schema = (
                "You are a precise SQL schema resolver. Given ONE SQL query and a DATABASE SCHEMA, "
                "identify all ACTUAL DATABASE COLUMNS referenced by the query and resolve each to its REAL BASE TABLE "
                "from the schema (not CTEs or aliases).\n"
                "Rules:\n"
                "- Consider FROM/JOIN clauses to map aliases to base tables.\n"
                "- Exclude CTE names and their columns; only return columns from REAL tables provided in schema.\n"
                "- Exclude computed expressions and SELECT aliases.\n"
                "- If unsure which base table, omit the column (do NOT guess).\n"
                "- Use the schema to validate table and column existence.\n"
                "Return ONLY JSON with this exact schema:\n"
                "{\n  \"columns\": [ { \"table\": \"RealTable\", \"column\": \"column_name\" } ]\n}"
            )
            resp = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt_schema},
                    {"role": "user", "content": f"DATABASE SCHEMA (JSON):\n{schema_context}\n\nSQL:\n{sql}"},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[len("```json"):]
            if content.endswith("```"):
                content = content[:-3]
            data = json.loads(content)
            raw_cols = data.get("columns", [])
            # Filter strictly to columns present in schema
            schema_tables = set(schema_info.keys())
            fq_cols: List[str] = []
            seen = set()
            for item in raw_cols:
                if not isinstance(item, dict):
                    continue
                t = (item.get("table") or "").strip()
                c = (item.get("column") or "").strip()
                if not t or not c:
                    continue
                if t not in schema_tables:
                    continue
                table_cols = {col["name"] for col in schema_info.get(t, [])}
                if c not in table_cols:
                    continue
                fq = f"{t}.{c}"
                if fq not in seen:
                    fq_cols.append(fq)
                    seen.add(fq)
            if fq_cols:
                print(f"🔍 LLM schema-resolved columns: {fq_cols}")
                return fq_cols
        except Exception as e:
            print(f"⚠️  LLM schema-resolve failed, will fallback: {e}")

    # Simpler prompt: just list database columns (may include alias-qualified).
    system_prompt = (
        "You are a precise SQL structure parser. Given one SQL query, identify all ACTUAL DATABASE COLUMNS "
        "that are referenced. Focus on the underlying data being accessed from tables, not aliases or computed columns. "
        "Include columns used in: SELECT, WHERE, GROUP BY, ORDER BY, HAVING clauses, and JOIN conditions. "
        "EXCLUDE: aliases (like 'AS col_name'), computed columns (like 'SUM(col)', 'COUNT(*)'), "
        "CTE names, subquery aliases, or temporary names. "
        "Include the base column name even if it has table prefixes (like 'table.column'). "
        "Deduplicate and preserve first-appearance order. Return ONLY JSON with the schema:\n"
        "{\"database_columns\": [\"...\", \"...\"]}"
    )

    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"SQL:\n{sql}"},
            ],
            temperature=1.0 if "o4-mini" in model or "o3-mini" in model or "o3" in model else 0.0,
        )
        content = resp.choices[0].message.content.strip()
        # Strip markdown fences if any
        if content.startswith("```json"):
            content = content[len("```json"):]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        parsed = json.loads(content)
        columns = parsed.get("database_columns", [])
        # Normalize to strings and strip whitespace
        out: List[str] = []
        seen = set()
        for c in columns:
            if not isinstance(c, str):
                continue
            cc = c.strip()
            if cc and cc not in seen:
                out.append(cc)
                seen.add(cc)
        
        if schema_info:
            # Qualify by unique table mapping (strip alias prefixes, bind to real table when unique)
            col_to_tables = {}
            for table_name, table_columns in schema_info.items():
                for col_info in table_columns:
                    nm = col_info['name']
                    col_to_tables.setdefault(nm, set()).add(table_name)

            qualified: List[str] = []
            seen = set()
            for token in out:
                base = token.split('.')[-1]
                candidates = col_to_tables.get(base, set())
                if len(candidates) == 1:
                    real_table = list(candidates)[0]
                    fq = f"{real_table}.{base}"
                    if fq not in seen:
                        qualified.append(fq)
                        seen.add(fq)
            print(f"🔍 LLM extracted columns: {out}")
            print(f"🔍 Filtered columns: {qualified}")
            return qualified
        return out
    except Exception:
        # Fallback regex - very conservative
        try:
            sql_no_cte = re.sub(r"^\s*with\s+.*?\)\s*select", " select", sql, flags=re.IGNORECASE | re.DOTALL)
        except Exception:
            sql_no_cte = sql
        
        # Pattern to match column references in various contexts
        patterns = [
            r"select\s+([^,\n]+?)(?:\s+from|\s+where|\s+group|\s+order|\s+having|$)",  # SELECT clause
            r"where\s+([^,\n]+?)(?:\s+group|\s+order|\s+having|$)",  # WHERE clause
            r"group\s+by\s+([^,\n]+?)(?:\s+order|\s+having|$)",  # GROUP BY clause
            r"order\s+by\s+([^,\n]+?)(?:\s+having|$)",  # ORDER BY clause
            r"having\s+([^,\n]+?)$",  # HAVING clause
        ]
        
        candidates = []
        for pattern in patterns:
            matches = re.findall(pattern, sql_no_cte, flags=re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by comma and extract individual column references
                parts = [p.strip() for p in match.split(',')]
                for part in parts:
                    # Extract column references (simple heuristic)
                    col_refs = re.findall(r'[`\"]?[\w\.\-]+[`\"]?', part)
                    candidates.extend(col_refs)
        
        # Clean up and deduplicate
        cleaned = []
        seen = set()
        for c in candidates:
            base = c.strip()
            if base and base not in seen:
                cleaned.append(base)
                seen.add(base)
        # If schema present, try to uniquely qualify
        if schema_info:
            col_to_tables = {}
            for table_name, table_columns in schema_info.items():
                for col_info in table_columns:
                    nm = col_info['name']
                    col_to_tables.setdefault(nm, set()).add(table_name)
            qualified = []
            seen_q = set()
            for token in cleaned:
                base = token.split('.')[-1]
                candidates = col_to_tables.get(base, set())
                if len(candidates) == 1:
                    real_table = list(candidates)[0]
                    fq = f"{real_table}.{base}"
                    if fq not in seen_q:
                        qualified.append(fq)
                        seen_q.add(fq)
            return qualified
        return cleaned


def compare_with_llm(correct_sql: str, agent_sql: str, model: str = "azure/gpt-4o") -> Dict[str, List[str]]:
    """Use LLM to directly compare two SQL queries focusing on database columns."""
    system_prompt = (
        "You are a precise SQL comparison expert. Given two SQL queries, identify SOURCE DATABASE TABLES and ACTUAL DATABASE COLUMNS "
        "that exist in the first SQL (correct) but are missing from the second SQL (agent-generated). "
        "Focus on: ONLY source database tables/views referenced in FROM/JOIN clauses (NOT CTEs, temp tables, or subquery aliases). "
        "ONLY include columns that exist in the source database tables, not computed columns, aliases, or expressions. "
        "CRITICAL: IGNORE ALL CTE names (defined in WITH clauses), subquery aliases, temporary table names, and computed columns. "
        "IGNORE: aliases (like 'AS col_name'), computed columns (like 'SUM(col)', 'COUNT(*)'), "
        "CTE names (like 'track_revenue', 'category_stats'), subquery aliases, or temporary names. "
        "Only report missing elements that represent actual SOURCE DATABASE access differences. "
        "Return ONLY JSON with the schema:\n"
        "{\"table_diff\": [\"...\", \"...\"], \"column_diff\": [\"...\", \"...\"]}\n"
        "If no differences are found, return empty lists."
    )

    try:
        # O4-Mini only supports temperature=1, other models can use temperature=0
        temperature = 1.0 if "o4-mini" in model else 0.0
        
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Correct SQL:\n{correct_sql}\n\nAgent SQL:\n{agent_sql}"},
            ],
            temperature=temperature,
        )
        content = resp.choices[0].message.content.strip()
        # Strip markdown fences if any
        if content.startswith("```json"):
            content = content[len("```json"):]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        parsed = json.loads(content)
        
        # Normalize and clean up the results
        table_diff = []
        column_diff = []
        
        for table in parsed.get("table_diff", []):
            if isinstance(table, str) and table.strip():
                table_diff.append(table.strip())
        
        for column in parsed.get("column_diff", []):
            if isinstance(column, str) and column.strip():
                column_diff.append(column.strip())
        
        return {
            "table_diff": table_diff,
            "column_diff": column_diff
        }
    except Exception as e:
        print(f"⚠️  LLM comparison failed: {e}")
        return {"table_diff": [], "column_diff": []}


def calculate_precision_recall_f1(correct_set: Set[str], agent_set: Set[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score for two sets."""
    if len(agent_set) == 0:
        precision = 0.0
    else:
        precision = len(correct_set & agent_set) / len(agent_set)
    
    if len(correct_set) == 0:
        recall = 0.0
    else:
        recall = len(correct_set & agent_set) / len(correct_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compare_with_parse(correct_sql: str, agent_sql: str, model: str = "azure/gpt-4o", instance_id: str = None) -> Dict[str, List[str]]:
    """Parse each SQL individually and compare the extracted tables and database columns."""
    try:
        # Get database schema if instance_id is provided
        schema_info = None
        if instance_id:
            try:
                schema_info = get_database_schema_for_instance(instance_id)
                print(f"📋 Loaded schema for {instance_id}: {list(schema_info.keys())}")
                print(f"🔍 Schema info type: {type(schema_info)}, length: {len(schema_info) if schema_info else 0}")
            except Exception as e:
                print(f"⚠️  Could not load schema for {instance_id}: {e}")
        
        # Extract tables and database columns from both SQLs
        correct_tables = set(extract_tables_with_llm(correct_sql, model, schema_info))
        agent_tables = set(extract_tables_with_llm(agent_sql, model, schema_info))
        
        correct_columns = set(extract_database_columns_with_llm(correct_sql, model, schema_info))
        agent_columns = set(extract_database_columns_with_llm(agent_sql, model, schema_info))
        
        # Normalize column names by removing table qualifiers for comparison
        def normalize_columns(columns):
            normalized = set()
            for col in columns:
                # Remove table qualifier (e.g., "Track.TrackId" -> "TrackId")
                if '.' in col:
                    normalized.add(col.split('.')[-1])
                else:
                    normalized.add(col)
            return normalized
        
        correct_columns_normalized = normalize_columns(correct_columns)
        agent_columns_normalized = normalize_columns(agent_columns)
        
        # Find differences
        table_diff = sorted(list(correct_tables - agent_tables))
        column_diff = sorted(list(correct_columns_normalized - agent_columns_normalized))
        
        # Calculate precision/recall/F1 for tables and columns
        table_metrics = calculate_precision_recall_f1(correct_tables, agent_tables)
        column_metrics = calculate_precision_recall_f1(correct_columns_normalized, agent_columns_normalized)
        
        return {
            "table_diff": table_diff,
            "column_diff": column_diff,
            "correct_tables": list(correct_tables),
            "agent_tables": list(agent_tables),
            "correct_columns": list(correct_columns),
            "agent_columns": list(agent_columns),
            "table_metrics": table_metrics,
            "column_metrics": column_metrics
        }
    except Exception as e:
        print(f"⚠️  Parse comparison failed: {e}")
        return {
            "table_diff": [], 
            "column_diff": [], 
            "correct_tables": [], 
            "agent_tables": [], 
            "correct_columns": [], 
            "agent_columns": [],
            "table_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "column_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }


def main():
    parser = argparse.ArgumentParser(description="Compare or extract SQL tables/columns focusing on real database objects")
    parser.add_argument('--mode', choices=['llm', 'parse', 'extract'], required=True, 
                       help='llm: direct LLM compare; parse: per-SQL extraction then compare; extract: extract tables/columns from one SQL')
    parser.add_argument('--correct-sql', type=str, required=True, 
                       help='Path to file containing the correct SQL (or the sole SQL when --mode extract)')
    parser.add_argument('--agent-sql', type=str, 
                       help='Path to file containing the agent-generated SQL (required for llm/parse modes)')
    parser.add_argument('--model', type=str, default='azure/gpt-4o', 
                       help='LLM model identifier for litellm')
    parser.add_argument('--instance-id', type=str, 
                       help='Instance ID to get database schema (for parse mode)')
    parser.add_argument('--output', type=str, 
                       help='Output file path (JSON format). If not specified, prints to stdout')
    parser.add_argument('--metrics-only', action='store_true',
                       help='Show only precision/recall/F1 metrics without detailed breakdown')
    
    args = parser.parse_args()
    
    try:
        # Read SQL files
        correct_sql = read_sql_from_file(args.correct_sql)
        agent_sql = read_sql_from_file(args.agent_sql) if args.agent_sql else None
        
        print(f"📖 Read correct SQL from: {args.correct_sql}")
        print(f"📖 Read agent SQL from: {args.agent_sql}")
        print(f"🔧 Using mode: {args.mode}")
        
        # Perform action based on mode
        if args.mode == 'llm':
            if not agent_sql:
                raise ValueError('--agent-sql is required for mode=llm')
            result = compare_with_llm(correct_sql, agent_sql, args.model)
        elif args.mode == 'parse':
            if not agent_sql:
                raise ValueError('--agent-sql is required for mode=parse')
            result = compare_with_parse(correct_sql, agent_sql, args.model, args.instance_id)
        else:  # extract
            schema_info = None
            if args.instance_id:
                try:
                    schema_info = get_database_schema_for_instance(args.instance_id)
                    print(f"📋 Loaded schema for {args.instance_id}: {list(schema_info.keys())}")
                    print(f"🔍 Schema info type: {type(schema_info)}, length: {len(schema_info) if schema_info else 0}")
                except Exception as e:
                    print(f"⚠️  Could not load schema for {args.instance_id}: {e}")
            tables = extract_tables_with_llm(correct_sql, args.model, schema_info)
            columns = extract_database_columns_with_llm(correct_sql, args.model, schema_info)
            result = {
                "tables": tables,
                "columns": columns,
            }
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"📄 Results written to: {args.output}")
        else:
            print("\n📊 Comparison Results:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Reporting
        if args.mode in ('llm','parse'):
            table_count = len(result['table_diff'])
            column_count = len(result['column_diff'])
            print(f"\n📈 Summary:")
            print(f"  Missing tables: {table_count}")
            print(f"  Missing database columns: {column_count}")
            
            if table_count > 0:
                print(f"  Missing tables: {result['table_diff']}")
            if column_count > 0:
                print(f"  Missing database columns: {result['column_diff']}")
            
            # Display precision/recall/F1 metrics (parse mode only)
            if 'table_metrics' in result and 'column_metrics' in result:
                print(f"\n📊 Precision/Recall/F1 Metrics:")
                print(f"  Tables:")
                print(f"    Precision: {result['table_metrics']['precision']:.3f}")
                print(f"    Recall:    {result['table_metrics']['recall']:.3f}")
                print(f"    F1:        {result['table_metrics']['f1']:.3f}")
                print(f"  Columns:")
                print(f"    Precision: {result['column_metrics']['precision']:.3f}")
                print(f"    Recall:    {result['column_metrics']['recall']:.3f}")
                print(f"    F1:        {result['column_metrics']['f1']:.3f}")
                
                # Show detailed breakdown unless --metrics-only is specified
                if not args.metrics_only:
                    if 'correct_tables' in result and 'agent_tables' in result:
                        correct_tables = set(result['correct_tables'])
                        agent_tables = set(result['agent_tables'])
                        print(f"\n🔍 Detailed Breakdown:")
                        print(f"  Tables:")
                        print(f"    Correct SQL tables: {sorted(correct_tables)}")
                        print(f"    Agent SQL tables:   {sorted(agent_tables)}")
                        print(f"    Intersection:       {sorted(correct_tables & agent_tables)}")
                        print(f"    Only in correct:    {sorted(correct_tables - agent_tables)}")
                        print(f"    Only in agent:      {sorted(agent_tables - correct_tables)}")
                    
                    if 'correct_columns' in result and 'agent_columns' in result:
                        correct_columns = set(result['correct_columns'])
                        agent_columns = set(result['agent_columns'])
                        print(f"  Columns:")
                        print(f"    Correct SQL columns: {sorted(correct_columns)}")
                        print(f"    Agent SQL columns:   {sorted(agent_columns)}")
                        print(f"    Intersection:        {sorted(correct_columns & agent_columns)}")
                        print(f"    Only in correct:     {sorted(correct_columns - agent_columns)}")
                        print(f"    Only in agent:       {sorted(agent_columns - correct_columns)}")
        else:
            # extract mode: brief summary
            print(f"\n📈 Extracted:")
            print(f"  Tables:  {result['tables']}")
            print(f"  Columns: {result['columns']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
