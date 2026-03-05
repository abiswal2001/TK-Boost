<p align="center">
  <img src="assets/logo.png" alt="TK-Boost Logo" width="250">
</p>

<h1 align="center">TK-Boost</h1>

<p align="center">
  <strong>Arming Data Agents with Tribal Knowledge</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.13521">
    <img src="https://img.shields.io/badge/arXiv-2602.13521-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://abiswal2001.github.io/TK-Boost/">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page">
  </a>
  <a href="https://abiswal2001.github.io/TK-Boost/blog.html">
    <img src="https://img.shields.io/badge/Blog-Post-green" alt="Blog Post">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

<p align="center">
  <em>Shubham Agarwal &middot; Asim Biswal &middot; Sepanta Zeighami &middot; Alvin Cheung &middot; Joseph Gonzalez &middot; Aditya G. Parameswaran</em><br>
  <em>UC Berkeley</em>
</p>

---

**TK-Boost** is a bolt-on framework for augmenting any NL2SQL agent with **tribal knowledge** — reusable corrective knowledge that fixes the agent's recurring misconceptions about real-world databases, learned from experience.

> NL2SQL agents already know how to write SQL. What they lack is experience using real databases. Tribal knowledge fills that gap.

### Key Results

| Agent | Spider2 (max gain) | BIRD (max gain) |
|:---|:---:|:---:|
| **GPT-4.1 Agent** | **+25.3%** | **+14.0%** |
| **ReFORCE** (bolt-on) | **+11.4%** | **+5.6%** |
| **Agentar-Scale-SQL-32B** (bolt-on) | **+10.2%** | **+3.6%** |

Spider2 gains are reported as the largest observed delta across SQLite, BigQuery, and Snowflake from the paper figures.

---

## Features

- **Bolt-On Design** — works with any NL2SQL agent, no retraining needed
- **Interpretable Knowledge** - visualize and understand the agent's data misconcpetions, and see how tribal knowledge helps.
- **Engine Agnostic** — SQLite, PostgresSQL, Snowflake, and BigQuery



---

## Quick Start

```bash
git clone https://github.com/abiswal2001/TK-Boost.git
cd TK-Boost
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

See TK-Boost in action yourself through the example below, or [visualize](#visualize-spider-2-tribal-knowledge) tribal knowledge from Spider-2.

Note: this example typically takes 5-10 minutes end-to-end (agent translation + knowledge generation + refinement).
```python
import tkboost
from tkboost import SQLiteExecutor, SQLAgent

# 1) Initialize credentials/model (uses env vars or explicit args)
tkboost.init(provider="auto", model="azure/gpt-5")

# 2) Create executor + ReAct SQL agent
executor = SQLiteExecutor("tkstore/example/Baseball.sqlite")
agent = SQLAgent()

# 3) Translate question -> draft SQL
draft = agent.translate(
    question="Compute the average career span in years for baseball players.",
    executor=executor,
    db_name="Baseball",
)

# 4) Generate tribal knowledge store
store = tkboost.generate(
    example_json="tkstore/example/example.json",  # local007
    store="tkstore/tkstore_example.csv",
    executor=executor,
    debug=True,
)

# 5) Refine same draft using tribal knowledge
result = tkboost.sql(
    draft=draft["sql"],
    executor=executor,
    store=store,
    db_name="Baseball",
)
```


Compare the refined SQL result to the initial draft result:

```python
from pathlib import Path

gold_sql = Path("tkstore/example/gold.sql").read_text(encoding="utf-8")

_, agent_rows = executor.execute(draft["sql"])
_, refined_rows = executor.execute(result["refined_sql"])
_, gold_rows = executor.execute(gold_sql)

print("agent:", agent_rows[:3])
print("refined:", refined_rows[:3])
print("gold:", gold_rows[:3])
```

Expected output (local007):
- `agent` is off (e.g., around `4.82`)
- `refined` fixes the result (e.g., around `4.92`)
- `gold` answer is `4.92375...`

### Visualize Spider-2 Tribal Knowledge

```python
from tkboost import TKStore

store = TKStore("tkstore/tkstore_example.csv")
store.visualize(port=8501)
```

Terminal one-liner:

```bash
python -c "from tkboost import TKStore; TKStore('tkstore/tkstore_example.csv').visualize()"
```

---

## Usage

Quickstart above covers end-to-end flow. This section includes supporting setup/reference info.

<details>
<summary><strong>Training data setup (`example.json` + files)</strong></summary>

Minimum required fields in each `example.json`:
- `example_id`
- `engine`
- `question`
- `gold_sql_path`

Common optional fields:
- `db_name` (preferred)
- `db_path` (needed for SQLite if executor is not passed)
- `agent_sql_path`, `gold_result_path`, `agent_result_path`
- `db_info_path`, `external_evidence_path`, `credential_path`

Minimal example:

```json
{
  "example_id": "local007",
  "db_name": "Baseball",
  "engine": "sqlite",
  "question": "Compute the average career span in years for baseball players.",
  "gold_sql_path": "gold.sql"
}
```

While generating knowledge for a directory of training examples (`examples_dir`), each example must be specified with its own `example_json`.
</details>

<details>
<summary><strong>Available SQL executors</strong></summary>

```python
from tkboost import SQLiteExecutor, SnowflakeExecutor, BigQueryExecutor, PostgresExecutor

sqlite_exec = SQLiteExecutor("/abs/path/to/database.sqlite")
snowflake_exec = SnowflakeExecutor("/abs/path/to/snowflake_credential.json")
bigquery_exec = BigQueryExecutor("/abs/path/to/service_account.json")
postgres_exec = PostgresExecutor("postgresql://user:pass@host:5432/dbname")
```

All executors implement `execute(sql)`.
</details>

<details>
<summary><strong>Debug traces</strong></summary>

Set `debug=True` in `tkboost.generate(...)` to persist intermediate artifacts (including `llm_interactions.json`) for each example.

These traces are used by the TKStore visualizer and are useful for diagnosing weak rules.
</details>

<details>
<summary><strong>More details</strong></summary>

- Full SDK reference: [`docs/API_SUMMARY.md`](docs/API_SUMMARY.md)
- Evaluation/repro instructions: [`evaluation/README.md`](evaluation/README.md)
</details>

---

## Configuration

<details>
<summary><strong>LLM Provider</strong></summary>

Edit `src/utils/auth.py`:

```python
# Azure OpenAI
os.environ["AZURE_API_KEY"] = "your-key"
os.environ["AZURE_API_BASE"] = "https://your-endpoint.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"

# Or OpenAI
os.environ["OPENAI_API_KEY"] = "your-key"
```

System prompts are in `src/agents/prompts.py`:
- `BASE_PROMPT` — SQLite instances
- `SNOWFLAKE_PROMPT` — Snowflake instances (syntax & case-sensitivity notes)
</details>

---

## Evaluation

See [`evaluation/README.md`](evaluation/README.md) for full setup and commands to reproduce Spider2 evaluation results.

---

## Citation

```bibtex
@article{agarwal2026tkboost,
  title={Arming Data Agents with Tribal Knowledge},
  author={Agarwal, Shubham and Biswal, Asim and Zeighami, Sepanta and Cheung, Alvin and Gonzalez, Joseph and Parameswaran, Aditya G.},
  journal={arXiv preprint arXiv:2602.13521},
  year={2026}
}
```

## License

MIT License — See [LICENSE](LICENSE) for details.
