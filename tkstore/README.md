# tkstore — Python usage

Small programmatic API for adding and retrieving indexed memories (LLM-backed tagging).

Usage (Python):
```python
from tkstore.harness import generate_rules_from_diff
from tkstore.tagger_index import MemoryRetriever

# Process an outputs directory end-to-end (tag + add to global store)
# Use the harness.run_diff_for_instance() function

# Then retrieve from the global index
mr = MemoryRetriever("/path/to/tkstore_bq.csv")
matches = mr.retrieve("SELECT order_id, AVG(payment_value) FROM orders_with_payment GROUP BY order_id;", generic_only=True)
print(matches)
```

Config:
- Defaults and index locations are in `tkstore/config.py`.
- Index files are named by database type: `tkstore_sqlite.csv`, `tkstore_bq.csv`, `tkstore_sf.csv`

