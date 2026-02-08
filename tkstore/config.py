"""Configuration for tkstore package paths.

Place to set default paths (leave None to require explicit CLI args).
Do NOT store secrets here; only filesystem path defaults.

Examples:
  JSONL_DEFAULT = '/path/to/spider2-lite.jsonl'
  DATA_BASE_FOLDER = '/path/to/data'
"""
from pathlib import Path

# Resolve paths relative to this config file's location
_CONFIG_DIR = Path(__file__).parent
_PROJECT_ROOT = _CONFIG_DIR.parent

JSONL_DEFAULT = str(_PROJECT_ROOT / 'data' / 'spider2-lite.jsonl')
DATA_BASE_FOLDER = 'data'

# TKStore index paths by database type
TKSTORE_SQLITE_PATH = str(_PROJECT_ROOT / 'tkstore' / 'tkstore_sqlite.csv')
TKSTORE_BQ_PATH = str(_PROJECT_ROOT / 'tkstore' / 'tkstore_bq.csv')
TKSTORE_SF_PATH = str(_PROJECT_ROOT / 'tkstore' / 'tkstore_sf.csv')
FAILED_INDEX_PATH = str(_PROJECT_ROOT / 'tkstore' / 'failed_memories.csv')

# Legacy alias for backwards compatibility (defaults to BQ)
MEMORY_INDEX_PATH = TKSTORE_BQ_PATH


