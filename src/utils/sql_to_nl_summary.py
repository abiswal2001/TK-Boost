#!/usr/bin/env python3
"""
Generate precise natural-language translations for GOLD SQLs.

Inputs:
- ground_truth_union/localXXX.sql             # GOLD SQL files
- gt_sql_metadata.csv                          # to select instance_ids
- spider2-lite.jsonl                           # to read user questions

Output:
- sql_nl_summaries.csv with columns:
  instance_id,user_query,gold_sql,nl_summary

Notes:
- Uses Azure LLM creds from sql_agent_runner.py
- Uses model azure/o3 by default
- Prints clean progress and totals
"""

import os
import csv
import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional

import litellm

# Inline Azure credentials (load from environment) and export env vars
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE", "https://east-docetl.openai.azure.com/")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")

if AZURE_API_KEY:
    os.environ["AZURE_API_KEY"] = AZURE_API_KEY
if AZURE_API_BASE:
    os.environ["AZURE_API_BASE"] = AZURE_API_BASE
if AZURE_API_VERSION:
    os.environ["AZURE_API_VERSION"] = AZURE_API_VERSION
# Some clients look for this name
if AZURE_API_KEY:
    os.environ.setdefault("AZURE_OPENAI_API_KEY", AZURE_API_KEY)

PROJECT_ROOT = Path(__file__).parent
GT_META_CSV = PROJECT_ROOT / "bq_gt_sql_metadata.csv"
QUESTIONS_JSONL = Path("questions_bq/gt_bq_questions.jsonl")
GROUND_TRUTH_DIR = PROJECT_ROOT / "correctbqsqls"
OUT_CSV_DEFAULT = PROJECT_ROOT / "bq_sql_nl_summaries.csv"
OUT_CSV_TAX = PROJECT_ROOT / "bq_sql_nl_summaries_taxonomy.csv"

MODEL = "azure/o3"


def load_instance_ids_from_meta(meta_csv: Path) -> List[str]:
    ids: List[str] = []
    with meta_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = (row.get("instance_id") or "").strip()
            if iid:
                ids.append(iid)
    return ids


def load_questions(jsonl_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            iid = str(obj.get("instance_id") or obj.get("id") or obj.get("tag") or "").strip()
            q = (obj.get("question") or obj.get("question_text") or obj.get("query") or "").strip()
            if iid and q:
                mapping[iid] = q
    return mapping


def read_gold_sql(iid: str, sql_dir: Path) -> Optional[str]:
    sql_path = sql_dir / f"{iid}.sql"
    if not sql_path.exists():
        return None
    try:
        return sql_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def build_system_prompt() -> str:
    return (
        "You are an expert SQL explainer. Translate SQL to precise natural language.\n"
        "Strict requirements:\n"
        "- Produce a clean, unambiguous, NON-LOSSY description that fully specifies the query logic.\n"
        "- No SQL keywords or syntax; no pseudo-SQL.\n"
        "- Do not use code blocks, backticks, or inline code.\n"
        "- Use clear English with exact conditions, filters, joins, and aggregations.\n"
        "- Spell out grouping, ordering, limits, window logic, and calculations.\n"
        "- Name tables and columns in plain English; avoid code formatting.\n"
        "- Include key constants, ranges, and join keys in words.\n"
        "- Final output must be a single paragraph, not a list, no headings.\n"
        "- Prioritize completeness over brevity; do not omit any detail present in the SQL.\n"
        "Coverage checklist (must address explicitly, in prose):\n"
        "  • Data sources used (all tables/derived datasets) and their roles.\n"
        "  • Joins between sources with the exact matching keys and join directions (inner/left/etc.).\n"
        "  • All filters with precise operators (equals/at least/greater than/between/contains/etc.) and constants/ranges.\n"
        "  • Grouping keys and every aggregation (sum/avg/count/etc.) with what is aggregated.\n"
        "  • HAVING-like conditions applied after aggregation.\n"
        "  • Window calculations: partitioning keys, ordering keys, and the frame (e.g., rows vs range and bounds).\n"
        "  • Set operations (union/intersect/except) and how duplicates are handled.\n"
        "  • DISTINCT/deduplication semantics and what defines uniqueness.\n"
        "  • Sorting in the final output (order, ascending/descending, nulls placement if implied).\n"
        "  • Row limiting or top-k logic, including tie-breaking order if present.\n"
        "  • Computed fields and formulas (including rounding/formatting), and date/time granularity or extraction.\n"
        "  • Final output columns and their meaning/units, in the order they appear.\n"
    )


def build_taxonomy_system_prompt() -> str:
    base = build_system_prompt()
    taxonomy = (
        "\n\nOutput a NON-LOSSY, fully specified explanation organized into these EXACT sections and labels:\n"
        "Tables and columns used:\n"
        "Joins performed:\n"
        "CTEs needed:\n"
        "Math logics:\n"
        "Other info:\n"
        "Guidance:\n"
        "- This taxonomy is ONLY a reformatting of the same complete content required above; do not omit or soften any detail.\n"
        "- Override: Instead of a single paragraph, structure the explanation strictly into the sections above.\n"
        "- No SQL keywords or syntax; no pseudo-SQL; no code/backticks.\n"
        "- Be specific about sources, join directions/keys, all filters (operators and constants), grouping and each aggregation, HAVING-like filters, window specs (partition/order/frame), set operations, DISTINCT semantics, ordering (asc/desc and nulls if implied), row limits/top-k and tie-breakers, computed fields/formulas/rounding, date/time granularity/extractions, and the final output columns and meaning.\n"
        "- Write continuous prose sentences (not bullets) inside each section, and use ONLY the exact section labels provided (no extra headings).\n"
        "- Keep each section concise but complete. If a section does not apply, write 'None'.\n"
        "- Place each detail in its appropriate section; avoid cross-section content and repetition.\n"
    )
    return base + taxonomy


def build_user_prompt(user_query: str, sql_text: str) -> str:
    return (
        "Context: The following is the user question for which the GOLD SQL was written.\n\n"
        f"User question:\n{user_query}\n\n"
        "Task: Provide a complete natural-language translation of the GOLD SQL so that an SQL-competent reader could reconstruct the same query without seeing code.\n"
        "Do not include any SQL keywords or code fragments.\n\n"
        f"GOLD SQL:\n{sql_text}\n"
    )


def llm_sql_to_nl(user_query: str, sql_text: str, taxonomise: bool = False) -> str:
    system_prompt = build_taxonomy_system_prompt() if taxonomise else build_system_prompt()
    user_prompt = build_user_prompt(user_query, sql_text)
    resp = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        drop_params=True,
    )
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate NL summaries for GOLD SQLs (vanilla or taxonomy mode)")
    parser.add_argument("--taxonomise", action="store_true", help="Output a structured, taxonomy-based non-lossy summary")
    args = parser.parse_args()

    print(f"📥 Loading inputs... model={MODEL} | taxonomy={'ON' if args.taxonomise else 'OFF'}")
    if not GT_META_CSV.exists():
        print(f"❌ Missing gt_sql_metadata.csv at {GT_META_CSV}")
        return
    if not QUESTIONS_JSONL.exists():
        print(f"❌ Missing questions JSONL at {QUESTIONS_JSONL}")
        return
    if not GROUND_TRUTH_DIR.is_dir():
        print(f"❌ Missing union SQL directory at {GROUND_TRUTH_DIR}")
        return

    instance_ids = load_instance_ids_from_meta(GT_META_CSV)
    questions = load_questions(QUESTIONS_JSONL)
    print(f"✅ {len(instance_ids)} instance_id(s) from metadata; {len(questions)} questions loaded")

    # Determine output path and load existing ids for checkpointing
    out_path = OUT_CSV_TAX if args.taxonomise else OUT_CSV_DEFAULT
    existing_ids: set[str] = set()
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "instance_id" in reader.fieldnames:
                    for row in reader:
                        iid = (row.get("instance_id") or "").strip()
                        if iid:
                            existing_ids.add(iid)
            print(f"🧷 Resuming: found {len(existing_ids)} existing row(s) in {out_path}")
        except Exception as e:
            print(f"⚠️  Could not read existing {out_path}: {e}")

    processed = 0
    skipped_missing_sql = 0
    skipped_missing_q = 0
    errors = 0

    # Prepare CSV append writer (write header only if file missing/empty)
    header = ["instance_id", "user_query", "gold_sql", "nl_summary"]
    header_needed = True
    if out_path.exists():
        try:
            header_needed = out_path.stat().st_size == 0
        except Exception:
            header_needed = False

    with out_path.open("a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=header)
        if header_needed:
            writer.writeheader()

        for iid in instance_ids:
            if iid in existing_ids:
                print(f"⏭️  Skipping already present in output: {iid}")
                continue
            sql_text = read_gold_sql(iid, GROUND_TRUTH_DIR)
            if not sql_text:
                print(f"❌ Skipping: no SQL at {GROUND_TRUTH_DIR / f'{iid}.sql'}")
                skipped_missing_sql += 1
                continue
            user_q = questions.get(iid)
            if not user_q:
                skipped_missing_q += 1
                continue
            # continue
            print(f"\n➡️  {iid}: translating GOLD SQL to NL...")
            print("────────────────────────────────────────────────────────")
            print("User question:")
            print(user_q)
            print("\nGOLD SQL:")
            print(sql_text)
            print("────────────────────────────────────────────────────────")
            print("🔄 Calling LLM" + (" (taxonomy mode)" if args.taxonomise else "") + "...")
            try:
                nl = llm_sql_to_nl(user_q, sql_text, taxonomise=args.taxonomise)
                print("\nNL summary:")
                print(nl)
                writer.writerow({
                    "instance_id": iid,
                    "user_query": user_q,
                    "gold_sql": sql_text,
                    "nl_summary": nl,
                })
                out_f.flush()
                processed += 1
                print(f"\n   ✅ done ({len(nl)} chars)")
            except Exception as e:
                errors += 1
                print(f"   ❌ LLM error: {e}")

    if processed == 0 and errors == 0 and skipped_missing_q == 0 and skipped_missing_sql == 0:
        print("\n⚠️  No new rows appended (all instances may already exist in output).")
    else:
        print(f"\n📝 Appended {processed} new row(s) to {out_path}")

    total = len(instance_ids)
    print("\n===== SUMMARY =====")
    print(f"Selected IDs          : {total}")
    print(f"Processed (translated): {processed}")
    print(f"Skipped (no SQL)      : {skipped_missing_sql}")
    print(f"Skipped (no question) : {skipped_missing_q}")
    print(f"Errors                : {errors}")


if __name__ == "__main__":
    main()