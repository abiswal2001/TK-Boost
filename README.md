<p align="center">
  <img src="raw/image (14).png" alt="TK-Boost Logo" width="200">
</p>

<h1 align="center">TK-Boost</h1>
<h3 align="center">Arming Data Agents with Tribal Knowledge</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2602.13521"><strong>Paper (arXiv)</strong></a> &nbsp;|&nbsp;
  <a href="https://skejriwal44.github.io/TK-Boost/"><strong>Project Page</strong></a> &nbsp;|&nbsp;
  <a href="https://skejriwal44.github.io/TK-Boost/blog.html"><strong>Blog Post</strong></a>
</p>

<p align="center">
  <em>Shubham Agarwal, Asim Biswal, Sepanta Zeighami, Alvin Cheung, Joseph Gonzalez, Aditya G. Parameswaran</em>
</p>

---

## Abstract

Natural language to SQL (NL2SQL) translation enables non-expert users to query relational databases through natural language. Recently, NL2SQL agents, powered by the reasoning capabilities of Large Language Models (LLMs), have significantly advanced NL2SQL translation. Nonetheless, NL2SQL agents still make mistakes when faced with large-scale real-world databases because they lack knowledge of how to correctly leverage the underlying data and form misconceptions about the data when querying it, leading to errors.

**TK-Boost** is a bolt-on framework for augmenting any NL2SQL agent with **tribal knowledge**: knowledge that corrects the agent's misconceptions in querying the database accumulated through experience using the database. TK-Boost first identifies the agent's misconceptions by analyzing its mistakes, generates tribal knowledge to address them, and indexes this knowledge with applicability conditions for accurate retrieval. When answering new queries, TK-Boost provides targeted feedback to the agent, resolving misconceptions during SQL generation.

### Key Results

| Benchmark | Improvement |
|---|---|
| Spider 2.0 | **+16.9%** execution accuracy |
| BIRD | **+13.7%** execution accuracy |
| ReFORCE (bolt-on) | **+11.4%** execution accuracy |
| Agentar-Scale SQL (bolt-on) | **+10.2%** execution accuracy |

---

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

## Demo

The **`demo.ipynb`** notebook is a short end-to-end walkthrough: it runs the SQL agent on a single SQLite instance (e.g. `local066`) with predicted CTE hints, then inspects the generated SQL and compares the agent result to ground truth. Use it to sanity-check your setup and see the runner in action before using the CLI.

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

## Validation-Only Mode (From Existing Outputs)

If you have already generated agent outputs and want to **only** run the refiner on existing SQL (without regenerating agent responses), use the `--validate-output` flag:

```bash
python src/agents/sql_agent_runner.py \
  --validate-output outputs/snowflake_norefiner \
  --jsonl-path data/spider2-lite.jsonl \
  --model azure/gpt-4.1 \
  -c data/contexts/predicted_cte_briefs_snowflake_azure_o3.csv \
  -t data/contexts/predicted_tablescols_snowflake_azure_o3.csv \
  --verbose
```

## Evaluation

Run evaluation against Spider2-lite gold results:

```bash
cd evaluation
python evals.py \
  --mode exec_result \
  --result_dir ../outputs/snowflake_validated \
  --gold_dir ../data/spider2/gold
```

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

## Citation

```bibtex
@article{agarwal2025arming,
  title={Arming Data Agents with Tribal Knowledge},
  author={Agarwal, Shubham and Biswal, Asim and Zeighami, Sepanta and Cheung, Alvin and Gonzalez, Joseph and Parameswaran, Aditya G.},
  journal={arXiv preprint arXiv:2602.13521},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.
