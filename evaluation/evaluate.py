"""
Evaluation runner for Spider2-lite outputs.

Usage examples:
- Evaluate entire outputs directory (per-instance subfolders with execution_result.csv):
  python evaluate.py --mode exec_result --result_dir outputs --gold_dir evaluation/gold

- Evaluate a single instance folder:
  python evaluate.py --mode exec_result --result_dir outputs/<instance_id>_<timestamp> --gold_dir evaluation/gold

Outputs:
- Prints one line per instance with score (0/1)
- Writes evals.csv with [instance_id, score]
- Writes correct_ids.csv with all instance_ids scoring 1
- Prints aggregate summary at end

Note: Code adapted from https://github.com/xlang-ai/Spider2/blob/main/spider2-lite/evaluation_suite/evaluate.py
"""

import json
import re
import pandas as pd
import math
import duckdb
from typing import List, Union
import os
import os.path as osp
import argparse
# from google.cloud import bigquery
import shutil
import sqlite3
from tqdm import tqdm
# import snowflake.connector
import logging

# Remove TeeOutput - let users redirect stdout themselves if needed
TOTAL_GB_PROCESSED = 0.0
NO_EXECUTION_RESULT_CSV_COUNT = 0


byte_output_dict = {}


def load_jsonl_to_dict(jsonl_file):
    """Load a JSONL file into a dict keyed by instance_id."""
    data_dict = {}
    with open(jsonl_file, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            instance_id = item['instance_id']
            data_dict[instance_id] = item
    return data_dict


def load_json_list_to_dict(json_file_path):
    """Load a JSON (list) file into a dict keyed by instance_id."""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
    data_dict = {item['instance_id']: item for item in data_list}
    return data_dict


def compare_multi_pandas_table(pred, multi_gold, multi_condition_cols=[], multi_ignore_order=False):
    # print('multi_condition_cols', multi_condition_cols)

    if multi_condition_cols == [] or multi_condition_cols == [[]] or multi_condition_cols == [None] or multi_condition_cols == None:
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(sublist, list) for sublist in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]
    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
            return 1
    return 0
        
    



def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
    """_summary_

    Args:
        pred (Dataframe): _description_
        gold (Dataframe): _description_
        condition_cols (list, optional): _description_. Defaults to [].
        ignore_order (bool, optional): _description_. Defaults to False.

    """
    # print('condition_cols', condition_cols)
    
    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        if ignore_order_:
            v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                    sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True
    
    if condition_cols != []:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    for _, gold in enumerate(t_gold_list):
        if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
            score = 0
        else:
            for j, pred in enumerate(t_pred_list):
                if vectors_match(gold, pred, ignore_order_=ignore_order):
                    break

    return score


def get_bigquery_sql_result(sql_query, is_save, save_dir=None, file_name="result.csv"):
    """
    is_save = True, output a 'result.csv'
    if_save = False, output a string
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bigquery_credential.json"
    client = bigquery.Client()


    try:
        query_job = client.query(sql_query)
        results = query_job.result().to_dataframe() 
        total_bytes_processed = query_job.total_bytes_processed
        gb_processed = total_bytes_processed / (1024 ** 3)
        print(f"GB processed: {gb_processed:.5f} GB")
        global TOTAL_GB_PROCESSED
        TOTAL_GB_PROCESSED += gb_processed
        print(f"Total GB processed: {TOTAL_GB_PROCESSED:.5f} GB")
        
         
        
        if results.empty:
            print("No data found for the specified query.")
            results.to_csv(os.path.join(save_dir, file_name), index=False)
            return False, None
        else:
            if is_save:
                results.to_csv(os.path.join(save_dir, file_name), index=False)
                return True, None
            else:
                value = results.iat[0, 0]
                return True, None
    except Exception as e:
        print("Error occurred while fetching data: ", e)  
        return False, str(e)
    return True, None


def get_snowflake_sql_result(sql_query, database_id, is_save, save_dir=None, file_name="result.csv"):
    """
    is_save = True, output a 'result.csv'
    if_save = False, output a string
    """
    snowflake_credential = json.load(open('snowflake_credential.json'))
    conn = snowflake.connector.connect(
        database=database_id,
        **snowflake_credential
    )
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        if df.empty:
            print("No data found for the specified query.")
            return False, None
        else:
            if is_save:
                df.to_csv(os.path.join(save_dir, file_name), index=False)
                return True, None
    except Exception as e:
        print("Error occurred while fetching data: ", e)  
        return False, str(e)


def get_sqlite_result(db_path, query, save_dir=None, file_name="result.csv", chunksize=500):
    conn = sqlite3.connect(db_path)
    memory_conn = sqlite3.connect(':memory:')

    conn.backup(memory_conn)
    
    try:
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, chunk in enumerate(pd.read_sql_query(query, memory_conn, chunksize=chunksize)):
                mode = 'a' if i > 0 else 'w'
                header = i == 0
                chunk.to_csv(os.path.join(save_dir, file_name), mode=mode, header=header, index=False)
        else:
            df = pd.read_sql_query(query, memory_conn)
            return True, df

    except Exception as e:
        print(f"An error occurred: {e}")
        return False, str(e)

    finally:
        memory_conn.close()
        conn.close()
    
    return True, None


def parse_instance_id_from_output_dir(dir_name: str) -> str:
    """Extract instance_id from '<instance_id>_YYYYMMDD_HHMMSS' folder name."""
    m = re.match(r"^(?P<id>.+)_(?P<date>\d{8})_(?P<time>\d{6})$", dir_name)
    if m:
        return m.group("id")
    return None


def _count_assistant_turns_from_raw_memories(raw_memories_path: str) -> Union[int, None]:
    """Return number of assistant messages in raw_memories.json; None if missing/unreadable."""
    try:
        if not os.path.exists(raw_memories_path):
            return None
        with open(raw_memories_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        messages = data.get('messages', [])
        return sum(1 for m in messages if isinstance(m, dict) and m.get('role') == 'assistant')
    except Exception:
        return None


def _print_df_preview(label: str, path: str, df: pd.DataFrame, max_rows: int = 3):
    try:
        rows, cols = df.shape
        print(f"{label}: {path} | shape={rows}x{cols}")
        if rows > 0:
            preview = df.head(max_rows)
            # Render a compact preview
            print(preview.to_string(index=False))
        else:
            print("(empty dataframe)")
    except Exception as e:
        print(f"(could not preview {label} at {path}: {e})")


def evaluate_spider2sql(args):
    """Evaluate outputs against gold CSVs, writing evals.csv and correct_ids.csv, and printing per-id score and aggregate."""
    global NO_EXECUTION_RESULT_CSV_COUNT
    mode = args.mode
    gold_result_dir = os.path.join(args.gold_dir, "exec_result")

    eval_standard_dict = load_jsonl_to_dict("spider2lite_eval.jsonl")

    gold_ids = list(eval_standard_dict.keys())
    pred_ids = []
    pred_paths = {}  # instance_id -> predicted CSV path
    instance_dir_paths = {}  # instance_id -> instance directory path
    missing_exec_ids = []  # instance_ids present as folders but missing execution_result.csv

    # Detect predictions
    if os.path.isdir(args.result_dir):
        # Case A: Single instance folder with execution_result.csv
        exec_csv_path = os.path.join(args.result_dir, "execution_result.csv")
        if os.path.exists(exec_csv_path):
            inst_id = parse_instance_id_from_output_dir(os.path.basename(args.result_dir)) or os.path.basename(args.result_dir)
            pred_ids.append(inst_id)
            pred_paths[inst_id] = exec_csv_path
            instance_dir_paths[inst_id] = args.result_dir
        else:
            # Case B: Parent outputs directory with many subfolders
            # import pdb; pdb.set_trace()
            # Collect all candidate subfolders per instance_id, then pick the latest by timestamp
            from collections import defaultdict
            candidates = defaultdict(list)  # inst_id -> list[(timestamp_int, sub_path, has_csv)]

            def _parse_ts_from_dirname(name: str):
                try:
                    m = re.match(r"^(.+)_([0-9]{8})_([0-9]{6})$", name)
                    if not m:
                        return None
                    date_s = m.group(2)
                    time_s = m.group(3)
                    # YYYYMMDDHHMMSS as int for ordering
                    return int(f"{date_s}{time_s}")
                except Exception:
                    return None

            for sub in os.listdir(args.result_dir):
                sub_path = os.path.join(args.result_dir, sub)
                if not os.path.isdir(sub_path):
                    continue
                inst_id = parse_instance_id_from_output_dir(sub)
                if not inst_id:
                    continue
                ts_val = _parse_ts_from_dirname(sub)
                exec_csv_path = os.path.join(sub_path, "execution_result.csv")
                has_csv = os.path.exists(exec_csv_path)
                candidates[inst_id].append((ts_val if ts_val is not None else -1, sub_path, has_csv))
                if not has_csv:
                    # Track as missing execution_result.csv
                    missing_exec_ids.append(inst_id)
                    NO_EXECUTION_RESULT_CSV_COUNT += 1
                    instance_dir_paths[inst_id] = sub_path

            # Choose latest per instance_id and delete older duplicates
            import shutil as _shutil
            for inst_id, items in candidates.items():
                # sort by timestamp, then by name to stabilize
                items_sorted = sorted(items, key=lambda x: (x[0], x[1]))
                # Prefer the latest with CSV; otherwise latest overall
                chosen_ts, chosen_dir, chosen_has_csv = None, None, None
                for ts, d, has_csv in reversed(items_sorted):
                    if has_csv:
                        chosen_ts, chosen_dir, chosen_has_csv = ts, d, has_csv
                        break
                if chosen_dir is None:
                    chosen_ts, chosen_dir, chosen_has_csv = items_sorted[-1]
                chosen_exec = os.path.join(chosen_dir, "execution_result.csv")
                pred_ids.append(inst_id)
                pred_paths[inst_id] = chosen_exec
                instance_dir_paths[inst_id] = chosen_dir
                # delete older ones
                older = [d for (_, d, _) in items_sorted if d != chosen_dir]
                if older:
                    print(f"🧹 {inst_id}: keeping latest '{os.path.basename(chosen_dir)}', deleting {len(older)} older run(s)")
                    for od in older:
                        try:
                            _shutil.rmtree(od)
                            print(f"   deleted: {od}")
                        except Exception as e:
                            print(f"   ⚠️ failed to delete {od}: {e}")
            # import pdb; pdb.set_trace()
            if not pred_ids:
                # Fallback: CSVs placed directly under result_dir
                for file in os.listdir(args.result_dir):
                    if file.endswith(".csv"):
                        pred_id = file.split(".")[0]
                        pred_ids.append(pred_id)
                        pred_paths[pred_id] = os.path.join(args.result_dir, file)
            # import pdb; pdb.set_trace()
            
    else:
        # Single file path
        if args.result_dir.endswith(".csv") and os.path.exists(args.result_dir):
            # If it's execution_result.csv, derive id from parent folder name
            base = os.path.basename(args.result_dir)
            parent = os.path.basename(os.path.dirname(args.result_dir))
            if base == "execution_result.csv":
                pred_id = parse_instance_id_from_output_dir(parent) or parent
                instance_dir_paths[pred_id] = os.path.dirname(args.result_dir)
            else:
                pred_id = os.path.splitext(base)[0]
            pred_ids = [pred_id]
            pred_paths[pred_id] = args.result_dir
            if pred_id not in instance_dir_paths:
                instance_dir_paths[pred_id] = os.path.dirname(args.result_dir)
        else:
            raise FileNotFoundError(f"Result path not found: {args.result_dir}")

    # Removed verbose debug print of eval standard keys to reduce noise
    gold_ids = list(eval_standard_dict.keys())
    eval_ids = list(set(gold_ids).intersection(pred_ids))
    eval_ids = sorted(eval_ids)

    output_results = []

    for id in eval_ids:
        # print(f"Evaluating {id}...")
        error_info = None
        score = 0
        assistant_turns = None
        try:
            pred_csv_path = pred_paths.get(id)
            instance_dir = instance_dir_paths.get(id) or (os.path.dirname(pred_csv_path) if pred_csv_path else None)
            if instance_dir:
                raw_memories_path = os.path.join(instance_dir, "raw_memories.json")
                assistant_turns = _count_assistant_turns_from_raw_memories(raw_memories_path)
            if not pred_csv_path or not os.path.exists(pred_csv_path):
                # No prediction CSV; still record assistant_turns if available
                output_results.append(
                    {
                        "instance_id": id,
                        "score": 0,
                        "score_final": 0,
                        "pred_sql": None,
                        "error_info": "missing execution_result.csv",
                        "assistant_turns": assistant_turns
                    }
                )
                # print("0")
                continue
            # Baseline prediction (.csv)
            pred_pd = pd.read_csv(pred_csv_path)
            # Final prediction prefers execution_result_final.csv if present; else fall back to baseline
            final_csv_path = os.path.join(instance_dir, "execution_result_final.csv") if instance_dir else None
            have_final = bool(final_csv_path and os.path.exists(final_csv_path))
            pred_pd_final = pd.read_csv(final_csv_path) if have_final else pred_pd

            # Find gold CSV(s) for this id
            if '_' in id:
                pattern = re.compile(rf'^{re.escape(id)}(_[a-z])?\.csv$')
            else:
                pattern = re.compile(rf'^{re.escape(id)}(_[a-z])?\.csv$')
            all_files = os.listdir(gold_result_dir)
            csv_files = [file for file in all_files if pattern.match(file)]
            csv_files = sorted(csv_files)
            if len(csv_files) == 0:
                continue

            if len(csv_files) == 1:
                gold_path = os.path.join(gold_result_dir, f"{id}.csv")
                gold_pd = pd.read_csv(gold_path)
                # Baseline score
                score = compare_pandas_table(
                    pred_pd,
                    gold_pd,
                    eval_standard_dict.get(id)['condition_cols'],
                    eval_standard_dict.get(id)['ignore_order']
                )
                # Final score (using final.csv when available; else baseline again)
                score_final = compare_pandas_table(
                    pred_pd_final,
                    gold_pd,
                    eval_standard_dict.get(id)['condition_cols'],
                    eval_standard_dict.get(id)['ignore_order']
                )
                # print(f"{score}")
            else:
                gold_paths = [os.path.join(gold_result_dir, file) for file in csv_files]
                gold_pds = [pd.read_csv(gp) for gp in gold_paths]
                # Baseline score
                score = compare_multi_pandas_table(
                    pred_pd,
                    gold_pds,
                    eval_standard_dict.get(id)['condition_cols'],
                    eval_standard_dict.get(id)['ignore_order']
                )
                # Final score
                score_final = compare_multi_pandas_table(
                    pred_pd_final,
                    gold_pds,
                    eval_standard_dict.get(id)['condition_cols'],
                    eval_standard_dict.get(id)['ignore_order']
                )
                # print(f"{score}")
        except Exception as e:
            error_info = str(e)
            score = 0
            score_final = 0
            # Still print the line with error score
            if pred_paths.get(id):
                pred_csv_path = pred_paths[id]
            else:
                pred_csv_path = f"<missing pred for {id}>"
            print(f"{pred_csv_path} | <gold error> | 0")
        
        output_results.append(
            {
                "instance_id": id,
                "score": score,
                "score_final": score_final,
                "pred_sql": None,
                "error_info": error_info,
                "assistant_turns": assistant_turns
            }
        )
    # Add missing execution_result cases (count as evaluated with score=0) if they exist in gold set and not already added
    for mid in sorted(set(missing_exec_ids).intersection(gold_ids)):
        if any(r["instance_id"] == mid for r in output_results):
            continue
        inst_dir = instance_dir_paths.get(mid)
        assistant_turns_mid = None
        if inst_dir:
            assistant_turns_mid = _count_assistant_turns_from_raw_memories(os.path.join(inst_dir, "raw_memories.json"))
        output_results.append({"instance_id": mid, "score": 0, "pred_sql": None, "error_info": "missing execution_result.csv", "assistant_turns": assistant_turns_mid})
        # print(f"Evaluating {mid}...")
        # print("0")

    rows = [{"instance_id": item['instance_id'], "score": item['score'], "score_final": item.get('score_final', 0), "assistant_turns": item.get('assistant_turns')} for item in output_results]
    df_rows = pd.DataFrame(rows)
    
    # Add tick/cross columns for readability
    try:
        df_rows["base"] = df_rows["score"].apply(lambda x: "✓" if int(x) == 1 else "✗")
        df_rows["final"] = df_rows["score_final"].apply(lambda x: "✓" if int(x) == 1 else "✗")
        df_rows["either"] = ((df_rows["score"].fillna(0).astype(int) | df_rows["score_final"].fillna(0).astype(int)) > 0).map({True: "✓", False: "✗"})
        df_rows["both"] = ((df_rows["score"].fillna(0).astype(int) & df_rows["score_final"].fillna(0).astype(int)) > 0).map({True: "✓", False: "✗"})
    except Exception:
        pass
    evals_csv = os.path.join(args.result_dir if os.path.isdir(args.result_dir) else os.path.dirname(args.result_dir), "evals.csv")
    df_rows.to_csv(evals_csv, index=False)
    
    # Aggregate summary
    total = len(rows)
    correct_base = sum(r["score"] for r in rows)
    ratio_base = (correct_base / total) if total else 0
    correct_final = sum(r.get("score_final", 0) for r in rows)
    ratio_final = (correct_final / total) if total else 0
    print(f"Aggregate (baseline .csv): {correct_base}/{total} = {ratio_base:.4f}")
    print(f"Aggregate (final .csv fallback .csv): {correct_final}/{total} = {ratio_final:.4f}")
    # Print assistant_turns average across instances with available raw_memories.json
    # Additional union/both/only stats
    base_only_ids = df_rows[(df_rows["score"].fillna(0) == 1) & (df_rows["score_final"].fillna(0) == 0)]["instance_id"].tolist()
    final_only_ids = df_rows[(df_rows["score"].fillna(0) == 0) & (df_rows["score_final"].fillna(0) == 1)]["instance_id"].tolist()
    both_ids = df_rows[(df_rows["score"].fillna(0) == 1) & (df_rows["score_final"].fillna(0) == 1)]["instance_id"].tolist()
    either_ids = df_rows[(df_rows["score"].fillna(0) == 1) | (df_rows["score_final"].fillna(0) == 1)]["instance_id"].tolist()
    # import pdb; pdb.set_trace()
    turn_series = pd.to_numeric(df_rows.get('assistant_turns'), errors='coerce')
    available = int(turn_series.count())
    if available:
        avg_turns = float(turn_series.mean())
        print(f"Average assistant_turns: {avg_turns:.2f} over {available}/{total} instances")
    else:
        print("Average assistant_turns: NA (no raw_memories.json found)")
    print(f"evals.csv: {evals_csv}")
    print(f"NO_EXECUTION_RESULT_CSV_COUNT: {NO_EXECUTION_RESULT_CSV_COUNT}")

    # Print union vs only sets
    try:
        print("\nBreakdown (IDs):")
        print(f"- Base only ({len(base_only_ids)}): {', '.join(base_only_ids)}")
        print(f"- Final only ({len(final_only_ids)}): {', '.join(final_only_ids)}")
        print(f"- Both ({len(both_ids)}): {', '.join(both_ids)}")
        print(f"- Either ({len(either_ids)}): {', '.join(either_ids)}")
    except Exception:
        pass

    # Write correct IDs list
    correct_ids = [r["instance_id"] for r in rows if r["score"] == 1]
    correct_csv = os.path.join(args.result_dir if os.path.isdir(args.result_dir) else os.path.dirname(args.result_dir), "correct_ids.csv")
    pd.DataFrame({"instance_id": correct_ids}).to_csv(correct_csv, index=False)
    print(f"correct_ids.csv: {correct_csv}")

    # Write correct IDs list for final
    correct_ids_final = [r["instance_id"] for r in rows if r.get("score_final", 0) == 1]
    correct_final_csv = os.path.join(args.result_dir if os.path.isdir(args.result_dir) else os.path.dirname(args.result_dir), "correct_ids_final.csv")
    pd.DataFrame({"instance_id": correct_ids_final}).to_csv(correct_final_csv, index=False)
    print(f"correct_ids_final.csv: {correct_final_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations for NLP models.")
    parser.add_argument("--mode", type=str, choices=["sql", "exec_result"], default='exec_result', help="Mode of submission results")
    parser.add_argument("--result_dir", type=str, default="outputs", help="Result directory")
    parser.add_argument("--gold_dir", type=str, default="gold", help="Result directory")
    parser.add_argument("--is_sql_debug", action="store_true", default=False)
    args = parser.parse_args()
    
    # if os.path.exists("temp"):
    #     shutil.rmtree("temp")
    # os.makedirs("temp")

    evaluate_spider2sql(args)
