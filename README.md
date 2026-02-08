# DBAgentMemory - SQL Agent Runner with CTE Refinement

A comprehensive framework for running SQL agents on multi-database benchmarks (SQLite, Snowflake, BigQuery) with optional CTE-based refinement, predicted schema hints, and external knowledge integration.

## Features

- **Multi-Database Support**: SQLite, Snowflake, and BigQuery with automatic engine detection
- **Predicted Hints**: Use LLM-generated table/column predictions and CTE briefs to guide agent exploration
- **External Knowledge**: Automatically inject relevant context from markdown files
- **CTE Refiner**: Optional refinement loop to iteratively improve SQL correctness
- **Refinement-Only Mode**: Run refiner on existing outputs without regenerating agent responses
- **Organized Output**: Structured directories with traces, queries, results, and ground truth
- **Parallel Execution**: Run multiple instances concurrently for efficient batch processing


## Installation

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Run SQL Agent (Single Instance)

Run the agent on a single instance without validation:

```bash
python src/agents/sql_agent_runner.py \
  --instance-id local066 \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  --out-base outputs/local_baseline \
  --verbose
```

### Run SQL Agent (Multiple Instances)

```bash
python src/agents/sql_agent_runner.py \
  --instance-id local066 \
  --instance-id local065 \
  --instance-id local022 \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  --out-base outputs/local_baseline \
  --verbose
```

### Run SQL Agent (All Instances from JSONL)

```bash
python src/agents/sql_agent_runner.py \
  --run-all-from-file \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  --out-base outputs/all_baseline \
  --verbose
```

### Run with Predicted Hints

Use predicted CTE briefs and table/column hints to guide the agent:

```bash
python src/agents/sql_agent_runner.py \
  --instance-id local066 \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  -c data/contexts/predicted_cte_briefs_local.csv \
  -t data/contexts/predicted_tablescols_local.csv \
  --out-base outputs/local_with_hints \
  --verbose
```

### Run Snowflake Instances

The agent automatically detects Snowflake instances (prefix `sf`) and uses the appropriate prompt and executor:

```bash
python src/agents/sql_agent_runner.py \
  --instance-id sf001 \
  --instance-id sf002 \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  -c data/contexts/predicted_cte_briefs_snowflake_azure_o3.csv \
  -t data/contexts/predicted_tablescols_snowflake_azure_o3.csv \
  --out-base outputs/snowflake_baseline \
  --verbose
```

**Note**: Snowflake credentials should be configured via environment variables or Snowflake config file.

### Run BigQuery Instances

The agent automatically detects BigQuery instances (prefix `bq` or `ga`) and uses the appropriate executor:

```bash
python src/agents/sql_agent_runner.py \
  --instance-id bq001 \
  --instance-id ga001 \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  -c data/contexts/predicted_cte_briefs_bigquery_azure_o3.csv \
  -t data/contexts/predicted_tablescols_bigquery_azure_o3.csv \
  --out-base outputs/bigquery_baseline \
  --verbose
```

**Note**: BigQuery credentials should be configured via Google Cloud SDK or service account JSON.

## Running with CTE Refiner

The CTE refiner iteratively checks and improves each CTE in the final SQL, then refines the final SELECT statement. Enable it with the `-v` or `--validate-cte` flag:

### Basic Validation

```bash
python src/agents/sql_agent_runner.py \
  --instance-id local066 \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  -c data/contexts/predicted_cte_briefs_local.csv \
  -t data/contexts/predicted_tablescols_local.csv \
  --validate-cte \
  --out-base outputs/local_validated \
  --verbose
```

**What happens:**
1. Agent generates initial SQL solution
2. SQL is parsed into CTEs and final SELECT
3. Each CTE is validated individually (max 25 turns per CTE)
4. If issues are found, refiner suggests fixes and agent revises
5. Final SELECT is validated with all CTEs in place
6. Validated SQL and results are saved

### Validation Output Files

When validation is enabled, additional files are created in the output directory:

```
outputs/local066_20251031_120000/
├── execution_query.sql                    # Original agent SQL
├── execution_result.csv                   # Original agent results
├── execution_query_validated.sql          # SQL after validation
├── execution_result_validated.csv         # Results after validation
├── refiner_cte1.json                    # Refiner verdict for first CTE
├── refiner_cte1_trace.txt               # Refinement trace for first CTE
├── refiner_cte2.json                    # Refiner verdict for second CTE
├── refiner_cte2_trace.txt               # Refinement trace for second CTE
├── refiner_final_select.json            # Refiner verdict for final SELECT
├── refiner_final_select_trace.txt       # Refinement trace for final SELECT
├── messages.json                          # Full conversation history
├── processed_trace.txt                    # Human-readable trace
├── gt_query.sql                           # Ground truth SQL
├── gt_result.csv                          # Ground truth results
└── gt_result.json                         # Ground truth results (JSON)
```

### Parallel Execution with Validation

Use the parallel runner to process multiple Snowflake instances with validation:

```bash
python scripts/run_snowflake_parallel.py \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  -c data/contexts/predicted_cte_briefs_snowflake_azure_o3.csv \
  -t data/contexts/predicted_tablescols_snowflake_azure_o3.csv \
  --out-base outputs/snowflake_validated \
  --workers 3 \
  --timeout 600 \
  --verbose
```

**Note**: The parallel runner in `scripts/run_snowflake_parallel.py` needs to be updated to support the `--validate-cte` flag if you want validation. Currently, it runs the agent without validation.

## Validation-Only Mode (From Existing Outputs)

If you have already generated agent outputs and want to **only** run the refiner on existing SQL (without regenerating agent responses), use the `--validate-output` flag:

### Run Refiner on Existing Output Directory

```bash
python src/agents/sql_agent_runner.py \
  --validate-output outputs/snowflake_norefiner \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  -c data/contexts/predicted_cte_briefs_snowflake_azure_o3.csv \
  -t data/contexts/predicted_tablescols_snowflake_azure_o3.csv \
  --verbose
```

**What happens:**
1. Scans `outputs/snowflake_norefiner/` for all instance directories
2. For each directory, loads the existing `execution_query.sql`
3. Runs the refiner loop (max 25 turns) starting from that SQL
4. Saves refined results in the same directories:
   - `execution_query_refined.sql`
   - `execution_result_validated.csv`
   - `execution_result_validated.json`
   - `refiner_*.json` and `refiner_*_trace.txt` files

**Use cases:**
- You ran the agent without validation and want to add validation later
- You want to re-run validation with different settings (e.g., different model)
- You want to validate a subset of outputs without re-generating them

### Example: Validate Specific Instances Only

First, manually select or filter the instance directories you want to validate, then run:

```bash
# Create a new directory with only the instances you want to validate
mkdir outputs/snowflake_subset
cp -r outputs/snowflake_norefiner/sf001_* outputs/snowflake_subset/
cp -r outputs/snowflake_norefiner/sf002_* outputs/snowflake_subset/

# Run validation on the subset
python src/agents/sql_agent_runner.py \
  --validate-output outputs/snowflake_subset \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  -c data/contexts/predicted_cte_briefs_snowflake_azure_o3.csv \
  -t data/contexts/predicted_tablescols_snowflake_azure_o3.csv \
  --verbose
```

## Generating Predicted Hints

### Generate Predicted Tables/Columns

```bash
# For SQLite instances (uses PRAGMA to explore schema)
python generate_predicted_tables_columns.py \
  --jsonl-path data/spider2-lite.jsonl \
  --taxonomy-csv data/contexts/sql_nl_summaries_taxonomy.csv \
  --out-csv data/contexts/predicted_tablescols_local.csv \
  --model azure/o3 \
  --engine sqlite \
  --verbose

# For Snowflake instances (uses precomputed schema contexts)
python generate_predicted_tables_columns.py \
  --jsonl-path data/spider2-lite.jsonl \
  --taxonomy-csv data/contexts/sql_nl_summaries_taxonomy.csv \
  --out-csv data/contexts/predicted_tablescols_snowflake_azure_o3.csv \
  --model azure/o3 \
  --engine snowflake \
  --all-snowflake-from-jsonl \
  --verbose

# For BigQuery instances (uses precomputed schema contexts)
python generate_predicted_tables_columns.py \
  --jsonl-path data/spider2-lite.jsonl \
  --taxonomy-csv data/contexts/sql_nl_summaries_taxonomy.csv \
  --out-csv data/contexts/predicted_tablescols_bigquery_azure_o3.csv \
  --model azure/o3 \
  --engine bigquery \
  --all-bigquery-from-jsonl \
  --verbose
```

### Generate Predicted CTE Briefs

```bash
# For SQLite instances
python generate_predicted_cte_briefs.py \
  --jsonl-path data/spider2-lite.jsonl \
  --taxonomy-csv data/contexts/sql_nl_summaries_taxonomy.csv \
  --analysis-csv data/contexts/sql_nl_summaries_taxonomy_analysis_of_summary_results.csv \
  --predicted-tables-cols-csv data/contexts/predicted_tablescols_local.csv \
  --out-csv data/contexts/predicted_cte_briefs_local.csv \
  --model azure/o3 \
  --restrict-to-predicted \
  --no-analysis-filter \
  --verbose

# For Snowflake instances (with external knowledge)
python generate_predicted_cte_briefs.py \
  --jsonl-path data/spider2-lite.jsonl \
  --taxonomy-csv data/contexts/sql_nl_summaries_taxonomy.csv \
  --analysis-csv data/contexts/sql_nl_summaries_taxonomy_analysis_of_summary_results.csv \
  --predicted-tables-cols-csv data/contexts/predicted_tablescols_snowflake_azure_o3.csv \
  --out-csv data/contexts/predicted_cte_briefs_snowflake_azure_o3.csv \
  --model azure/o3 \
  --include-external-knowledge \
  --external-knowledge-root data/spider2 \
  --snowflake-ids-only \
  --restrict-to-predicted \
  --no-analysis-filter \
  --verbose

# For BigQuery instances (with external knowledge)
python generate_predicted_cte_briefs.py \
  --jsonl-path data/spider2-lite.jsonl \
  --taxonomy-csv data/contexts/sql_nl_summaries_taxonomy.csv \
  --analysis-csv data/contexts/sql_nl_summaries_taxonomy_analysis_of_summary_results.csv \
  --predicted-tables-cols-csv data/contexts/predicted_tablescols_bigquery_azure_o3.csv \
  --out-csv data/contexts/predicted_cte_briefs_bigquery_azure_o3.csv \
  --model azure/o3 \
  --include-external-knowledge \
  --external-knowledge-root data/spider2 \
  --bigquery-ids-only \
  --restrict-to-predicted \
  --no-analysis-filter \
  --verbose
```

### Generate Schema Contexts (Snowflake/BigQuery)

For Snowflake and BigQuery, you need to precompute compressed schema context files:

```bash
# Snowflake: Creates data/sf_schemas/*.txt
python scripts/precompute_snowflake_db_contexts.py

# BigQuery: Creates data/bq_schemas/*.txt
python scripts/create_bq_schema_contexts.py
```

These scripts parse DDL files and create compressed, human-readable schema summaries that are injected into the agent's context.

## Configuration

### LLM Provider

Edit `src/utils/auth.py` to configure Azure OpenAI or OpenAI credentials:

```python
# Azure OpenAI
os.environ["AZURE_API_KEY"] = "your-key"
os.environ["AZURE_API_BASE"] = "https://your-endpoint.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"

# Or OpenAI
os.environ["OPENAI_API_KEY"] = "your-key"
```

### Database Credentials

**Snowflake**: Configure via `~/.snowflake/config` or environment variables:
```bash
export SNOWFLAKE_ACCOUNT="your_account"
export SNOWFLAKE_USER="your_user"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_WAREHOUSE="your_warehouse"
```

**BigQuery**: Configure via Google Cloud SDK:
```bash
gcloud auth application-default login
# Or use service account JSON
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**SQLite**: Databases are read from `data/spider2/` directory structure.

### System Prompts

System prompts are defined in `src/agents/prompts.py`:
- `BASE_PROMPT`: For SQLite instances
- `SNOWFLAKE_PROMPT`: For Snowflake instances (includes syntax guidance and case sensitivity notes)

BigQuery instances currently use `BASE_PROMPT` but can be extended with a `BIGQUERY_PROMPT` if needed.

## Evaluation

Run evaluation against Spider2-lite gold results:

```bash
cd evaluation
python evals.py \
  --mode exec_result \
  --result_dir ../outputs/snowflake_validated \
  --gold_dir ../data/spider2/gold
```

**Output:**
- `evals.csv`: Score (0/1) per instance
- `correct_ids.csv`: List of correct instance IDs
- Summary statistics printed to console

## Advanced Usage

### Custom Refiner Settings

The refiner uses `max_turns=25` by default (reduced from 40 for efficiency). To adjust, edit the calls to `refiner_run()` in `src/agents/sql_agent_runner.py`.

### Debugging

Enable verbose output to see detailed agent reasoning and SQL execution:

```bash
python src/agents/sql_agent_runner.py \
  --instance-id local066 \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  --verbose
```

Traces are saved in `processed_trace.txt` in each output directory.

### Resume from Existing Messages

To continue from an existing conversation (not yet fully supported in cleaned runner):

```bash
python old_code/sql_agent_runner.py \
  --instance_id local066 \
  --load-trace outputs/local066_20250909_123456/raw_memories.json \
  --user-feedback
```

## Troubleshooting

### Snowflake Connection Issues
- Verify credentials in `~/.snowflake/config`
- Check network connectivity to Snowflake
- Ensure warehouse is running

### BigQuery Permission Errors
- Run `gcloud auth application-default login`
- Verify service account has BigQuery permissions
- Check project ID in credentials

### LLM Rate Limits
- Reduce `--workers` in parallel runner
- Add retry logic with exponential backoff
- Use a different model tier

### Empty Validation Results
- Check that `execution_query.sql` exists in output directories
- Verify instance IDs match between JSONL and output directories
- Ensure predicted hints CSVs contain the instances you're validating

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.
