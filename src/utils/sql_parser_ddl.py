#!/usr/bin/env python3
"""
SQL Table/Column Parser using DDL CSV file
Uses schema-aware parsing to avoid CTEs and aliases
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import litellm

def load_ddl_schema(ddl_csv_path: str) -> dict:
    """Load database schema from DDL CSV file."""
    try:
        df = pd.read_csv(ddl_csv_path)
        schema = {}
        
        # Group by table_name and create column lists
        for table_name, group in df.groupby('table_name'):
            columns = group['column_name'].tolist()
            schema[table_name] = columns
            
        return schema
    except Exception as e:
        print(f"❌ Error loading DDL CSV: {e}")
        return {}

def extract_tables_with_llm(sql: str, model: str = "anthropic/claude-3-5-sonnet-20241022", schema_info: dict = None) -> list:
    """Use LLM to extract base table/view identifiers from SQL with schema awareness."""
    
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
    
    user_message = f"SQL Query:\n{sql}"
    
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in result:
            json_start = result.find("```json") + 7
            json_end = result.find("```", json_start)
            result = result[json_start:json_end].strip()
        elif "```" in result:
            json_start = result.find("```") + 3
            json_end = result.find("```", json_start)
            result = result[json_start:json_end].strip()
        
        parsed_result = json.loads(result)
        tables = parsed_result.get("tables", [])
        
        # Manual filtering against schema
        if schema_info:
            schema_tables = set(schema_info.keys())
            filtered_tables = [t for t in tables if t in schema_tables]
            print(f"🔍 LLM extracted: {tables}")
            print(f"📋 Schema tables: {list(schema_tables)}")
            print(f"✅ Filtered result: {filtered_tables}")
            return filtered_tables
        
        return tables
        
    except Exception as e:
        print(f"⚠️  LLM extraction failed: {e}")
        return []

def extract_columns_with_llm(sql: str, model: str = "anthropic/claude-3-5-sonnet-20241022", schema_info: dict = None) -> list:
    """Use LLM to extract column identifiers from SQL with schema awareness."""
    
    schema_context = ""
    if schema_info:
        # Create a flat list of all columns for filtering
        all_columns = set()
        for table_cols in schema_info.values():
            all_columns.update(table_cols)
        schema_context = f"\n\nDATABASE COLUMNS:\n{list(all_columns)}\n\n"
    
    system_prompt = (
        "You are a precise SQL structure parser. Given one SQL query, identify ALL COLUMN NAMES "
        "that are referenced in SELECT, WHERE, GROUP BY, ORDER BY, HAVING clauses. "
        "\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Include both qualified (table.column) and unqualified (column) references\n"
        "2. EXCLUDE table names, function names, and SQL keywords\n"
        "3. ONLY INCLUDE COLUMNS THAT APPEAR IN THE DATABASE SCHEMA BELOW\n"
        "\n"
        f"{schema_context}"
        "Return ONLY JSON with the schema:\n"
        "{\"columns\": [\"...\", \"...\"]}"
    )
    
    user_message = f"SQL Query:\n{sql}"
    
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in result:
            json_start = result.find("```json") + 7
            json_end = result.find("```", json_start)
            result = result[json_start:json_end].strip()
        elif "```" in result:
            json_start = result.find("```") + 3
            json_end = result.find("```", json_start)
            result = result[json_start:json_end].strip()
        
        parsed_result = json.loads(result)
        columns = parsed_result.get("columns", [])
        
        # Manual filtering against schema
        if schema_info:
            all_columns = set()
            for table_cols in schema_info.values():
                all_columns.update(table_cols)
            
            # Check both qualified and unqualified column names
            filtered_columns = []
            for col in columns:
                # Check unqualified column name
                unqualified = col.split('.')[-1] if '.' in col else col
                if unqualified in all_columns:
                    filtered_columns.append(col)
            
            print(f"🔍 LLM extracted: {columns}")
            print(f"📋 Schema columns: {list(all_columns)[:10]}...")  # Show first 10
            print(f"✅ Filtered result: {filtered_columns}")
            return filtered_columns
        
        return columns
        
    except Exception as e:
        print(f"⚠️  LLM extraction failed: {e}")
        return []

def parse_sql_with_ddl(sql_file: str, ddl_csv_path: str, model: str = "anthropic/claude-3-5-sonnet-20241022") -> dict:
    """Parse SQL file using DDL CSV for schema-aware extraction."""
    
    # Load SQL
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql = f.read()
    
    # Load schema from DDL CSV
    schema_info = load_ddl_schema(ddl_csv_path)
    if not schema_info:
        print("❌ No schema loaded, proceeding without filtering")
    
    print(f"📊 Loaded schema with {len(schema_info)} tables")
    
    # Extract tables and columns
    print("\n🔍 Extracting tables...")
    tables = extract_tables_with_llm(sql, model, schema_info)
    
    print("\n🔍 Extracting columns...")
    columns = extract_columns_with_llm(sql, model, schema_info)
    
    result = {
        "tables": tables,
        "columns": columns,
        "schema_tables": list(schema_info.keys()) if schema_info else [],
        "total_schema_columns": sum(len(cols) for cols in schema_info.values()) if schema_info else 0
    }
    
    return result

def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='SQL Parser using DDL CSV')
    parser.add_argument('--sql_file', type=str, required=True, help='Path to SQL file')
    parser.add_argument('--ddl_csv', type=str, required=True, help='Path to DDL CSV file')
    parser.add_argument('--model', type=str, default='anthropic/claude-3-5-sonnet-20241022', 
                       help='LLM model to use')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Parse SQL with DDL
    result = parse_sql_with_ddl(args.sql_file, args.ddl_csv, args.model)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    else:
        print(f"\n📊 Results:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
