#!/usr/bin/env python3
"""
Table Extractor via LLM (GPT-4o)
================================

Scans output folders, reads the final SQL, asks an LLM to extract all base tables/views
referenced in the SQL (FROM/JOIN), and writes a CSV with columns:
  - instance_id
  - sql
  - tables (JSON list)

Additionally, computes pairwise table-overlap across instances and writes a second CSV with:
  - id_a, id_b, overlap_count, tables_a, tables_b, overlap_tables, jaccard

Sources for SQL per output folder (first available):
  1) execution_query.sql
  2) execution_result.json -> final_solution

Selection of folders:
  - --folder: process this specific output folder
  - --instance-ids: process most recent folders for these instances under --outputs-dir
  - default: process all folders under --outputs-dir that contain execution_result.csv

Examples:
  python table_extractor.py --outputs-dir outputs --out-csv table_usage.csv
  python table_extractor.py --folder outputs/local066_20250916_123456 --out-csv table_usage.csv
  python table_extractor.py --instance-ids local066,local065 --out-csv table_usage.csv

Requirements:
  - litellm configured for Azure/OpenAI; model default: azure/gpt-4o
  - Uses the same Azure settings as sql_agent_runner.py
"""

import os
import re
import json
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import litellm


# ----------------- Configuration -----------------
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE", "https://east-docetl.openai.azure.com/")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")

# Set up environment (match sql_agent_runner.py)
if AZURE_API_KEY:
    os.environ["AZURE_API_KEY"] = AZURE_API_KEY
if AZURE_API_BASE:
    os.environ["AZURE_API_BASE"] = AZURE_API_BASE
if AZURE_API_VERSION:
    os.environ["AZURE_API_VERSION"] = AZURE_API_VERSION


def parse_instance_id_from_output_dir(dir_name: str) -> Optional[str]:
    """Extract instance_id from '<id>_YYYYMMDD_HHMMSS' folder names."""
    m = re.match(r"^(?P<id>.+)_(?P<date>\d{8})_(?P<time>\d{6})$", dir_name)
    return m.group("id") if m else None


def find_latest_dirs_for_instances(outputs_dir: Path, instance_ids: List[str]) -> List[Path]:
    """For each instance id, pick the most recent outputs/<id>_* folder (by timestamp suffix)."""
    selected: List[Path] = []
    index: Dict[str, List[Tuple[str, Path]]] = {}
    for entry in outputs_dir.iterdir():
        if not entry.is_dir():
            continue
        inst = parse_instance_id_from_output_dir(entry.name)
        if inst and inst in instance_ids:
            index.setdefault(inst, []).append((entry.name, entry))
    for inst in instance_ids:
        if inst in index:
            # sort by folder name lexicographically -> timestamp order works with YYYYMMDD_HHMMSS
            entries = sorted(index[inst], key=lambda x: x[0], reverse=True)
            selected.append(entries[0][1])
    return selected


def discover_all_result_dirs(outputs_dir: Path) -> List[Path]:
    """Return all outputs subdirs that contain an execution_result.csv file."""
    result = []
    for entry in outputs_dir.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "execution_result.csv").exists():
            result.append(entry)
    return sorted(result)


def load_sql_from_output_dir(folder: Path) -> Optional[str]:
    """Load SQL from execution_query.sql or fallback to execution_result.json['final_solution']."""
    sql_path = folder / "execution_query.sql"
    if sql_path.exists():
        try:
            with open(sql_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    json_path = folder / "execution_result.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sql = data.get("final_solution")
                if isinstance(sql, str) and sql.strip():
                    return sql.strip()
        except Exception:
            pass
    return None


def extract_tables_with_llm(sql: str, model: str = "azure/gpt-4o") -> List[str]:
    """Use LLM to extract base table/view identifiers from SQL.

    Returns a list of unique table identifiers (as they appear in the SQL, including quotes/backticks),
    excluding CTE names and aliases. Order should reflect first appearance.
    """
    system_prompt = (
        "You are a precise SQL structure parser. Given one SQL query, identify all base tables or views "
        "that are referenced in FROM or JOIN clauses. Do NOT include: column names, CTE names defined in "
        "the WITH clause, subquery aliases, or temp names. Keep identifiers exactly as written in the SQL "
        "(including quotes/backticks and qualification like schema.table or project.dataset.table). "
        "Deduplicate and preserve first-appearance order. Return ONLY JSON with the schema:\n"
        "{\"tables\": [\"...\", \"...\"]}"
    )

    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"SQL:\n{sql}"},
            ],
            temperature=0.0,
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
        return out
    except Exception:
        # Fallback regex: very conservative heuristic
        try:
            # Remove WITH CTE definitions to avoid capturing CTE names
            sql_no_cte = re.sub(r"^\s*with\s+.*?\)\s*select", " select", sql, flags=re.IGNORECASE | re.DOTALL)
        except Exception:
            sql_no_cte = sql
        pattern = r"(?:from|join)\s+([`\"]?[\w\.\-]+[`\"]?)"
        candidates = re.findall(pattern, sql_no_cte, flags=re.IGNORECASE)
        # Strip trailing alias tokens if any (split by whitespace)
        cleaned = []
        seen = set()
        for c in candidates:
            base = c.strip()
            # If there are spaces (rare due to regex), keep first token
            base = base.split()[0]
            if base and base not in seen:
                cleaned.append(base)
                seen.add(base)
        return cleaned


def collect_targets(args) -> List[Path]:
    outputs_dir = Path(args.outputs_dir)
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists() or not folder.is_dir():
            print(f"❌ Folder not found or not a directory: {folder}")
            return []
        return [folder]
    if args.instance_ids:
        ids = [x.strip() for x in args.instance_ids.split(',') if x.strip()]
        return find_latest_dirs_for_instances(outputs_dir, ids)
    # default: all with execution_result.csv
    return discover_all_result_dirs(outputs_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract tables used by SQL via GPT-4o and save CSV")
    parser.add_argument('--outputs-dir', type=str, default='outputs', help='Base outputs directory to scan')
    parser.add_argument('--folder', type=str, help='Specific output folder to process')
    parser.add_argument('--instance-ids', type=str, help='Comma-separated instance IDs to process (picks latest folder per ID)')
    parser.add_argument('--out-csv', type=str, default='table_usage.csv', help='Destination CSV path')
    parser.add_argument('--pairs-csv', type=str, default='table_overlap_pairs.csv', help='Destination CSV for pairwise overlaps')
    parser.add_argument('--top-k', type=int, default=200, help='Limit to top-K pairs by overlap (after filters)')
    parser.add_argument('--min-overlap', type=int, default=1, help='Minimum number of overlapping tables to include a pair')
    parser.add_argument('--correct-ids', type=str, default='outputs_vanilla/correct_ids.csv', help='Path to correct_ids.csv to mark success flags')
    parser.add_argument('--model', type=str, default='azure/gpt-4o', help='LLM model identifier for litellm')
    parser.add_argument('--questions-jsonl', type=str, default='questions_bq/spider2-lite.jsonl', help='Questions JSONL path to enrich db per instance')
    args = parser.parse_args()

    out_path = Path(args.out_csv)

    rows = []
    computed_usage = False

    if out_path.exists():
        # Skip recomputing table usage; load existing file and enrich with db
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                has_db_col = 'db' in (reader.fieldnames or [])
                for r in reader:
                    rows.append({
                        'instance_id': r.get('instance_id',''),
                        'db': r.get('db','') if has_db_col else '',
                        'sql': r.get('sql',''),
                        'tables': r.get('tables','[]')
                    })
            # Enrich db from questions JSONL when missing
            db_map = {}
            try:
                qpath = Path(args.questions_jsonl)
                if qpath.exists():
                    with open(qpath, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            iid = (obj.get('instance_id') or obj.get('id') or obj.get('tag') or '').strip()
                            dbn = (obj.get('db') or obj.get('database') or '').strip()
                            if iid and dbn:
                                db_map[iid] = dbn
            except Exception as e:
                print(f"⚠️  Failed to read questions JSONL for db enrichment: {e}")
            updated = False
            for r in rows:
                if not (r.get('db') or '').strip():
                    iid = (r.get('instance_id') or '').strip()
                    if iid in db_map:
                        r['db'] = db_map[iid]
                        updated = True
            if updated or not has_db_col:
                try:
                    with open(out_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=['instance_id','db','sql','tables'])
                        writer.writeheader()
                        for r in rows:
                            writer.writerow(r)
                    print(f"🛠️  Updated {out_path} with 'db' column/enrichment for {len(rows)} rows.")
                except Exception as e:
                    print(f"⚠️  Failed to rewrite {out_path} with db column: {e}")
            print(f"ℹ️  Using existing table usage from {out_path} ({len(rows)} rows).")
        except Exception as e:
            print(f"❌ Failed to read existing {out_path}: {e}")
            return
    else:
        targets = collect_targets(args)
        if not targets:
            print("❌ No target folders found.")
            return

        total = len(targets)
        print(f"🔎 Processing {total} folder(s)...")
        for idx, folder in enumerate(targets, start=1):
            inst = parse_instance_id_from_output_dir(folder.name) or "unknown"
            sql = load_sql_from_output_dir(folder)
            if not sql:
                print(f"[{idx}/{total}] ⚠️  No SQL found in {folder}")
                continue
            try:
                tables = extract_tables_with_llm(sql, model=args.model)
            except Exception as e:
                print(f"[{idx}/{total}] ❌ LLM error in {folder}: {e}")
                tables = []
            print(f"[{idx}/{total}] ✅ {folder.name} | tables: {tables}")
            rows.append({
                'instance_id': inst,
                'sql': sql,
                'tables': json.dumps(tables, ensure_ascii=False)
            })

        # Enrich with db from questions JSONL
        db_map = {}
        try:
            qpath = Path(args.questions_jsonl)
            if qpath.exists():
                with open(qpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        iid = (obj.get('instance_id') or obj.get('id') or obj.get('tag') or '').strip()
                        dbn = (obj.get('db') or obj.get('database') or '').strip()
                        if iid and dbn:
                            db_map[iid] = dbn
        except Exception as e:
            print(f"⚠️  Failed to read questions JSONL for db enrichment: {e}")
        for r in rows:
            iid = (r.get('instance_id') or '').strip()
            r['db'] = db_map.get(iid, '')

        # Write usage CSV only if we computed it now
        try:
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['instance_id', 'db', 'sql', 'tables'])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            computed_usage = True
            print(f"\n📄 Wrote {len(rows)} records to {out_path}")
        except Exception as e:
            print(f"❌ Failed to write CSV {out_path}: {e}")

    # Compute pairwise table overlaps
    try:
        # Prepare mapping: instance_id -> set(tables) and db
        id_to_tables = {}
        id_to_sql = {}
        id_to_db = {}
        for r in rows:
            inst = r['instance_id']
            id_to_sql[inst] = r['sql']
            id_to_db[inst] = (r.get('db') or '').strip()
            try:
                tbls = json.loads(r['tables'])
                if isinstance(tbls, list):
                    id_to_tables[inst] = set([t for t in tbls if isinstance(t, str)])
                else:
                    id_to_tables[inst] = set()
            except Exception:
                id_to_tables[inst] = set()

        instances = sorted(id_to_tables.keys())

        # Load success set (0/1 flags) from correct_ids.csv
        def load_success_set(outputs_dir: Path, override_path: Optional[str]) -> set:
            # Priority: explicit --correct-ids; else outputs_dir/correct_ids.csv; else try sibling outputs_vanilla/correct_ids.csv
            candidates = []
            if override_path:
                candidates.append(Path(override_path))
            candidates.append(outputs_dir / 'correct_ids.csv')
            # heuristic fallback
            candidates.append(Path('outputs_vanilla') / 'correct_ids.csv')
            success = set()
            for p in candidates:
                if p and p.exists():
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                iid = (row.get('instance_id') or '').strip()
                                if iid:
                                    success.add(iid)
                        print(f"ℹ️  Loaded success ids from {p} ({len(success)} ids)")
                        break
                    except Exception as e:
                        print(f"⚠️  Failed reading {p}: {e}")
            return success

        success_ids = load_success_set(Path(args.outputs_dir), args.correct_ids)
        pair_rows = []
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                ida = instances[i]
                idb = instances[j]
                set_a = id_to_tables.get(ida, set())
                set_b = id_to_tables.get(idb, set())
                if not set_a or not set_b:
                    continue
                # Restrict to same DB when available
                dba = id_to_db.get(ida, '')
                dbb = id_to_db.get(idb, '')
                if dba and dbb and dba != dbb:
                    continue
                overlap = sorted(list(set_a & set_b))
                if len(overlap) < args.min_overlap:
                    continue
                union = set_a | set_b
                jacc = (len(overlap) / len(union)) if union else 0.0
                pair_rows.append({
                    'id_a': ida,
                    'id_b': idb,
                    'success_a': 1 if ida in success_ids else 0,
                    'success_b': 1 if idb in success_ids else 0,
                    'overlap_count': len(overlap),
                    'overlap_tables': json.dumps(overlap, ensure_ascii=False),
                    'tables_a': json.dumps(sorted(list(set_a)), ensure_ascii=False),
                    'tables_b': json.dumps(sorted(list(set_b)), ensure_ascii=False),
                    'jaccard': f"{jacc:.6f}",
                })

        # Sort pairs: primary by overlap_count desc, secondary by jaccard desc, then ids
        pair_rows.sort(key=lambda r: (-r['overlap_count'], -float(r['jaccard']), r['id_a'], r['id_b']))
        if args.top_k and args.top_k > 0:
            pair_rows = pair_rows[:args.top_k]

        pairs_path = Path(args.pairs_csv)
        with open(pairs_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['id_a', 'id_b', 'success_a', 'success_b', 'overlap_count', 'jaccard', 'overlap_tables', 'tables_a', 'tables_b']
            )
            writer.writeheader()
            for pr in pair_rows:
                writer.writerow(pr)
        print(f"📄 Wrote {len(pair_rows)} pairwise rows to {pairs_path}")
    except Exception as e:
        print(f"⚠️  Failed computing/writing pairwise overlaps: {e}")


if __name__ == "__main__":
    main()


