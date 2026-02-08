#!/usr/bin/env python3
"""
Create compressed schema context files for BigQuery databases.
Similar to Snowflake schema creation, but for BigQuery instances.
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

def load_bq_instances() -> List[Dict]:
    """Load all BigQuery instances from spider2-lite.jsonl"""
    instances = []
    with open("data/spider2-lite.jsonl", "r") as f:
        for line in f:
            inst = json.loads(line)
            instance_id = inst.get("instance_id", "")
            if instance_id.startswith("bq") or instance_id.startswith("ga"):
                instances.append(inst)
    return instances

def find_ddl_files(instance_id: str, db_name: str) -> List[Path]:
    """Find all DDL.csv files for a given instance"""
    instance_dir = Path("data/spider2") / instance_id / db_name
    if not instance_dir.exists():
        return []
    
    ddl_files = []
    # Search recursively for DDL.csv files
    for ddl_path in instance_dir.rglob("DDL.csv"):
        ddl_files.append(ddl_path)
    
    return ddl_files

def parse_ddl_csv(ddl_path: Path) -> List[Dict]:
    """Parse a BigQuery DDL.csv file and return table information
    
    BigQuery DDL.csv format:
    table_name,ddl
    table1,"CREATE TABLE `project.dataset.table` (col1 TYPE, col2 TYPE, ...)"
    """
    tables = []
    try:
        with open(ddl_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                table_name = row.get("table_name", "")
                ddl_statement = row.get("ddl", "")
                
                if not table_name or not ddl_statement:
                    continue
                
                # Extract full qualified name from CREATE TABLE statement
                # Format: CREATE TABLE `project.dataset.table`
                import re
                fq_match = re.search(r'CREATE TABLE `([^`]+)`', ddl_statement)
                full_table_name = fq_match.group(1) if fq_match else table_name
                
                # Parse columns from CREATE TABLE statement
                # Extract content between ( and )
                match = re.search(r'\((.*)\)', ddl_statement, re.DOTALL)
                if not match:
                    continue
                
                columns_text = match.group(1)
                # Split by lines and parse each column definition
                for line in columns_text.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('--'):
                        continue
                    
                    # Parse column definition: col_name TYPE [OPTIONS(...)]
                    # Remove trailing comma
                    line = line.rstrip(',')
                    
                    # Extract column name and type (first two tokens)
                    parts = line.split(None, 2)  # Split on whitespace, max 3 parts
                    if len(parts) >= 2:
                        col_name = parts[0].strip('`')
                        col_type = parts[1]
                        
                        tables.append({
                            "table_name": table_name,  # Short name for grouping
                            "full_table_name": full_table_name,  # Full project.dataset.table
                            "column_name": col_name,
                            "data_type": col_type,
                            "is_nullable": "",  # Not easily extractable from CREATE TABLE
                        })
    except Exception as e:
        print(f"⚠️  Error parsing {ddl_path}: {e}")
    
    return tables

def compress_schema(tables: List[Dict], dataset_name: str) -> str:
    """Compress table schema information into a concise text format matching Snowflake style"""
    if not tables:
        return ""
    
    # Group by table
    table_groups = defaultdict(list)
    for row in tables:
        table_name = row["table_name"]
        if table_name:
            table_groups[table_name].append(row)
    
    # Group tables with same schema (common for dated tables like ga_sessions_20210101, etc.)
    # Also store full table names
    schema_groups = defaultdict(list)
    full_names_map = {}  # Map short name to full name
    for table_name, columns in table_groups.items():
        # Create schema signature
        col_sig = tuple(sorted((c["column_name"], c["data_type"]) for c in columns))
        schema_groups[col_sig].append(table_name)
        # Store full name (should be same for all columns of a table)
        if columns and "full_table_name" in columns[0]:
            full_names_map[table_name] = columns[0]["full_table_name"]
        else:
            full_names_map[table_name] = f"{dataset_name}.{table_name}"
    
    # Build compressed output
    lines = []
    lines.append(f"[DATASET={dataset_name}]")
    
    for col_sig, table_names in sorted(schema_groups.items(), key=lambda x: x[1][0]):
        # Get columns for this schema
        columns = table_groups[table_names[0]]
        
        # Group columns by type
        num_types = {"INT64", "FLOAT64", "NUMERIC", "BIGNUMERIC", "DECIMAL"}
        str_types = {"STRING", "BYTES"}
        time_types = {"TIMESTAMP", "DATE", "TIME", "DATETIME"}
        bool_types = {"BOOL", "BOOLEAN"}
        
        num_cols = []
        str_cols = []
        time_cols = []
        bool_cols = []
        other_cols = []
        
        for col in columns:
            col_name = col["column_name"]
            col_type = col["data_type"]
            
            if col_type in num_types:
                num_cols.append(col_name)
            elif col_type in str_types:
                str_cols.append(col_name)
            elif col_type in time_types:
                time_cols.append(col_name)
            elif col_type in bool_types:
                bool_cols.append(col_name)
            else:
                other_cols.append(col_name)
        
        # Format table name(s) using full qualified names
        if len(table_names) == 1:
            table_header = full_names_map.get(table_names[0], f"{dataset_name}.{table_names[0]}")
        else:
            # Group by common prefix
            table_names_sorted = sorted(table_names)
            # Get full names
            full_names = [full_names_map.get(t, f"{dataset_name}.{t}") for t in table_names_sorted]
            
            if len(table_names_sorted) > 5:
                # Show pattern for many similar tables using full name
                # Extract common parts from full names
                first_full = full_names[0]
                # Get project.dataset prefix
                parts = first_full.split('.')
                if len(parts) >= 2:
                    prefix_base = '.'.join(parts[:2])  # project.dataset
                    short_prefix = os.path.commonprefix([t for t in table_names_sorted])
                    table_header = f"{prefix_base}.{short_prefix}* ({len(table_names)} tables)"
                else:
                    table_header = f"{first_full} + {len(table_names)-1} similar"
            else:
                # Show all using short names with full prefix
                first_full = full_names[0]
                parts = first_full.split('.')
                if len(parts) >= 2:
                    prefix_base = '.'.join(parts[:2])
                    table_header = f"{prefix_base}.[{' | '.join(table_names_sorted)}]"
                else:
                    table_header = f"[{' | '.join(full_names)}]"
        
        lines.append(table_header)
        
        # Add column groups
        if num_cols:
            lines.append(f"  NUM cols: {', '.join(num_cols)}")
        if str_cols:
            lines.append(f"  STR cols: {', '.join(str_cols)}")
        if time_cols:
            lines.append(f"  TIME cols: {', '.join(time_cols)}")
        if bool_cols:
            lines.append(f"  BOOL cols: {', '.join(bool_cols)}")
        if other_cols:
            lines.append(f"  OTHER cols: {', '.join(other_cols)}")
        
        lines.append("")  # Blank line between tables
    
    return "\n".join(lines)

def main():
    print("🔍 Loading BigQuery instances...")
    instances = load_bq_instances()
    print(f"✅ Found {len(instances)} BigQuery instances")
    
    # Create output directory
    output_dir = Path("data/bq_schemas")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Track databases we've processed
    processed_dbs = set()
    db_to_instances = defaultdict(list)
    
    # Group instances by database
    for inst in instances:
        db_name = inst.get("db", "")
        if db_name:
            db_to_instances[db_name].append(inst["instance_id"])
    
    print(f"\n📊 Processing {len(db_to_instances)} unique databases...")
    
    success_count = 0
    skip_count = 0
    
    for db_name, instance_ids in sorted(db_to_instances.items()):
        if db_name in processed_dbs:
            continue
        
        print(f"\n🔄 Processing database: {db_name}")
        print(f"   Used by {len(instance_ids)} instance(s): {', '.join(instance_ids[:3])}")
        if len(instance_ids) > 3:
            print(f"   ... and {len(instance_ids) - 3} more")
        
        # Find DDL files from any instance using this database
        all_tables = []
        found_ddl = False
        
        for instance_id in instance_ids:
            ddl_files = find_ddl_files(instance_id, db_name)
            if ddl_files:
                found_ddl = True
                for ddl_path in ddl_files:
                    tables = parse_ddl_csv(ddl_path)
                    all_tables.extend(tables)
                    print(f"   📄 Found DDL: {ddl_path.relative_to('data/spider2')}")
                break  # Found DDL for this database
        
        if not found_ddl:
            print(f"   ⚠️  No DDL files found, skipping")
            skip_count += 1
            continue
        
        # Deduplicate tables (same table might appear in multiple DDL files)
        unique_tables = []
        seen = set()
        for table in all_tables:
            key = (table["table_name"], table["column_name"])
            if key not in seen:
                seen.add(key)
                unique_tables.append(table)
        
        # Compress and save
        compressed = compress_schema(unique_tables, db_name)
        if compressed:
            output_path = output_dir / f"{db_name}.txt"
            output_path.write_text(compressed, encoding="utf-8")
            print(f"   ✅ Saved: {output_path.name} ({len(unique_tables)} columns)")
            success_count += 1
        
        processed_dbs.add(db_name)
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Created: {success_count} schema files")
    print(f"⚠️  Skipped: {skip_count} (no DDL found)")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
