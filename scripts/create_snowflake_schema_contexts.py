#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Reuse analyzer helpers
from scripts.build_snowflake_context_analyzer import (
    stream_csv_for_create_blocks,
    parse_create_table_block,
    collapse_numeric_suffix_tables,
    normalize_table_name_for_context,
    signature_for_cols,
    build_compact_table_block,
    label_for_shards,
    approx_token_count,
)

PER_FILE_TIMEOUT_SECS = 30  # honored inside stream_csv_for_create_blocks
MAX_TABLES_PER_SCHEMA = 25


def find_snowflake_instances(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.is_dir() and p.name.startswith(("sf", "sf_", "sf-bq", "sf_bq"))]


def collect_db_to_ddl_paths(spider2_root: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for inst_dir in find_snowflake_instances(spider2_root):
        for db_dir in [d for d in inst_dir.iterdir() if d.is_dir()]:
            db_name = db_dir.name
            ddls = list(db_dir.rglob("DDL.csv"))
            if ddls:
                mapping.setdefault(db_name, []).extend(ddls)
    # de-duplicate paths
    for k, v in mapping.items():
        mapping[k] = sorted(list(set(v)))
    return mapping


def build_db_context(db_name: str, ddl_paths: List[Path], user_question: str = "") -> Tuple[str, List[str]]:
    flags: List[str] = []
    if not ddl_paths:
        return "", ["missing_ddl"]
    # keyword tokens (unused for now; could add filtering similar to analyzer)
    kw: Set[str] = set()
    # Group tables by schema (folder under the DB)
    groups: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    for p in ddl_paths:
        # schema = parent folder under DB
        try:
            schema = p.parent.name
        except Exception:
            schema = "UNKNOWN_SCHEMA"
        blocks = stream_csv_for_create_blocks(p)
        if not blocks:
            flags.append(f"no_blocks:{str(p)}")
            continue
        added_tables = 0
        for blk in blocks:
            tname, cols = parse_create_table_block(blk)
            if not tname or not cols:
                continue
            groups.setdefault(schema, {})
            if tname not in groups[schema]:
                groups[schema][tname] = cols
                added_tables += 1
            if len(groups[schema]) >= MAX_TABLES_PER_SCHEMA:
                flags.append(f"truncated_tables:{db_name}.{schema}")
                break

    parts: List[str] = []
    for schema, tbls in sorted(groups.items()):
        parts.append(f"[DB={db_name} | SCHEMA={schema}]")
        collapsed = collapse_numeric_suffix_tables(tbls)
        items = []
        for tname, data in sorted(collapsed.items()):
            base_name = normalize_table_name_for_context(db_name, schema, tname)
            items.append((base_name, data[0], data[1]))
        sig_map: Dict[Tuple[Tuple[str, str], ...], List[Tuple[str, List[Tuple[str, str]], List[str]]]] = {}
        for tname, cols, shards in items:
            sig = signature_for_cols(cols)
            sig_map.setdefault(sig, []).append((tname, cols, shards))
        for sig, entries in sig_map.items():
            rep_tname, rep_cols, rep_shards = entries[0]
            fq = f"{db_name}.{schema}.{rep_tname}"
            parts.append(build_compact_table_block(fq, rep_cols, kw))
            if rep_shards:
                parts.append(f"  {label_for_shards(rep_shards)} {', '.join(rep_shards)}")
            if len(entries) > 1:
                others = [t for (t, _, _) in entries[1:]]
                # compress numbered series in table names for readability
                from scripts.build_snowflake_context_analyzer import compress_numbered_series
                others_comp = compress_numbered_series(others)
                parts.append(f"  OTHER TABLES WITH IDENTICAL SCHEMA: {', '.join(sorted(others_comp))}")
            parts.append("")
    text = "\n".join(parts).strip() + "\n"
    return text, flags


def main():
    ap = argparse.ArgumentParser(description="Precompute compressed contexts per Snowflake DB")
    ap.add_argument("--spider2-root", default="data/spider2", help="Root path containing instance folders")
    ap.add_argument("--out-dir", default="data/sf_schemas", help="Output directory for db contexts (.txt only)")
    ap.add_argument("--db", action="append", default=[], help="Specific DB name(s) to precompute; can repeat")
    args = ap.parse_args()

    spider2_root = Path(args.spider2_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_map = collect_db_to_ddl_paths(spider2_root)
    targets = args.db or sorted(db_map.keys())
    if not targets:
        print("⚠️  No Snowflake DBs found under", spider2_root)
        return

    for db_name in targets:
        paths = db_map.get(db_name, [])
        if not paths:
            print(f"⚠️  {db_name}: no DDL paths found")
            continue
        print(f"🛠️  Building context for DB={db_name} (DDL files={len(paths)})")
        ctx, flags = build_db_context(db_name, paths)
        txt_path = out_dir / f"{db_name}.txt"
        txt_path.write_text(ctx, encoding="utf-8")
        print(f"✅ Wrote {txt_path} (chars={len(ctx)}, ~tokens={approx_token_count(ctx)}) flags={','.join(flags) if flags else '-'}")


if __name__ == "__main__":
    main()


