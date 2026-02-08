# Evaluation

Evaluation suite for Spider2-lite outputs.

## Setup

Before running evaluations, you need to set up the required files from the Spider2 repository:

### 1. Gold Results

Copy the gold evaluation results from the Spider2 repository:

```bash
# Clone Spider2 repo if you haven't already
git clone https://github.com/xlang-ai/Spider2.git

# Copy gold directory to this evaluation folder
cp -r Spider2/spider2-lite/evaluation_suite/gold evaluation/
```

The `gold/` directory should contain:
- `exec_result/*.csv` - Gold execution results for each instance
- `exec_result/*.sql` - Gold SQL queries (optional, for reference)

### 2. Evaluation Standards File

Copy the evaluation standards JSONL file:

```bash
cp Spider2/spider2-lite/evaluation_suite/spider2lite_eval.jsonl evaluation/
```

This file contains evaluation metadata for each instance (condition columns, ignore order flags, etc.).

## Usage

### Evaluate All Outputs

```bash
python evaluation/evaluate.py \
  --mode exec_result \
  --result_dir outputs \
  --gold_dir evaluation/gold
```

### Evaluate Single Instance

```bash
python evaluation/evaluate.py \
  --mode exec_result \
  --result_dir outputs/local066_20250101_120000 \
  --gold_dir evaluation/gold
```

## Output Files

The evaluation script creates:
- `evals.csv` - Detailed results for each instance (instance_id, score, score_final, assistant_turns, base, final, either, both)
- `correct_ids.csv` - List of instances with perfect scores
- `correct_ids_final.csv` - List of instances with perfect final scores

## Scoring

- **score** (base): Evaluation of `execution_result.csv`
- **score_final**: Evaluation of `execution_result_final.csv` (falls back to base if not present)
- **either**: Instance scored correctly in at least one version
- **both**: Instance scored correctly in both versions

Scores are 0 (incorrect) or 1 (correct).
