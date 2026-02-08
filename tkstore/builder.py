from pathlib import Path
from typing import List, Optional, Dict, Any

from .tagger_index import MemoryIndex, generate_tagged_memories_json


def add_memory(
    instance_id: str,
    clean_summary: str,
    database_memories: List[str],
    generic_memories: List[str],
    out_dir: str,
    model: str = "azure/o4-mini",
    verbose: bool = True,
    multiturn: bool = True,
) -> Optional[Dict[str, Any]]:
    """Create tagged JSON for memories and append them to the tkstore index CSV.

    This function calls the LLM tagger and then uses `MemoryIndex.append_tagged`
    to persist the results. Returns the parsed tagged JSON (or None on failure).
    """
    tagged = generate_tagged_memories_json(
        instance_id=instance_id,
        user_query="",
        db_name=None,
        gold_sql="",
        agent_sql="",
        clean_summary=clean_summary,
        database_memories=database_memories,
        generic_memories=generic_memories,
        evidence=None,
        minimal_required_edits=None,
        model=model,
        verbose=verbose,
        multiturn=multiturn,
    )

    if tagged is None:
        return None

    idx = MemoryIndex(out_dir)
    # ensure out_dir exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    idx.append_tagged(tagged, out_dir, str(instance_id), verbose=verbose)
    return tagged


class MemoryBuilder:
    def __init__(self, out_dir: str, model: str = "azure/o4-mini", verbose: bool = True):
        self.out_dir = out_dir
        self.model = model
        self.verbose = verbose

    def add(self, instance_id: str, clean_summary: str, database_memories: List[str], generic_memories: List[str]):
        return add_memory(instance_id, clean_summary, database_memories, generic_memories, self.out_dir, model=self.model, verbose=self.verbose)


