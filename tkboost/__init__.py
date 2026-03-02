"""Public high-level API for TK-Boost quickstart usage.

Example:
    import tkboost
    tkboost.init(provider="auto", api_key="...")
    tkboost.generate(example_json="path/to/example.json", store_path="tkstore/my_store.csv")
"""

import os
from typing import Any, Dict, List, Optional, Union

from tkstore.builder import build_knowledge_from_example, build_knowledge_from_examples_dir


_STATE: Dict[str, Any] = {
    "provider": "auto",
    "model": "gpt-4o-mini",
    "draft_sql_model": None,
}


def init(
    provider: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    model: Optional[str] = None,
    draft_sql_model: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize LLM auth/config for TK-Boost.

    Supports provider:
      - "auto" (default): choose OpenAI if OPENAI_API_KEY is set, else Azure if Azure keys are set
      - "openai"
      - "azure"
    """
    provider = (provider or "auto").lower().strip()
    if provider not in {"auto", "openai", "azure"}:
        raise ValueError("provider must be one of: auto, openai, azure")

    selected = provider
    if selected == "auto":
        if os.environ.get("OPENAI_API_KEY") or api_key:
            selected = "openai"
        elif os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or azure_api_key:
            selected = "azure"
        else:
            selected = "openai"

    if selected == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_API_BASE"] = base_url
        default_model = "gpt-4o-mini"
    else:
        if azure_api_key:
            os.environ["AZURE_API_KEY"] = azure_api_key
            os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
        elif api_key:
            # convenience alias
            os.environ["AZURE_API_KEY"] = api_key
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
        if azure_base_url:
            os.environ["AZURE_API_BASE"] = azure_base_url
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_base_url
        elif base_url:
            os.environ["AZURE_API_BASE"] = base_url
            os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
        if api_version:
            os.environ["AZURE_API_VERSION"] = api_version
        default_model = "azure/o4-mini"

    _STATE["provider"] = selected
    _STATE["model"] = model or default_model
    _STATE["draft_sql_model"] = draft_sql_model

    return {
        "provider": _STATE["provider"],
        "model": _STATE["model"],
        "draft_sql_model": _STATE["draft_sql_model"],
    }


def generate(
    example_json: Optional[str] = None,
    examples_dir: Optional[str] = None,
    store_path: Optional[str] = None,
    model: Optional[str] = None,
    draft_sql_model: Optional[str] = None,
    max_turns: int = 6,
    verbose: bool = True,
    hint: Optional[str] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate tribal knowledge and write/append to a store CSV.

    - If store_path exists, rows are appended.
    - If store_path does not exist, it is created.
    - Provide exactly one of: example_json or examples_dir.
    """
    if bool(example_json) == bool(examples_dir):
        raise ValueError("Provide exactly one of example_json or examples_dir")

    effective_model = model or _STATE.get("model") or "gpt-4o-mini"
    effective_draft = draft_sql_model if draft_sql_model is not None else _STATE.get("draft_sql_model")

    if example_json:
        return build_knowledge_from_example(
            example_json_path=example_json,
            index_path=store_path,
            model=effective_model,
            draft_sql_model=effective_draft,
            max_turns=max_turns,
            verbose=verbose,
            hint=hint,
        )

    return build_knowledge_from_examples_dir(
        examples_root=examples_dir or "",
        index_path=store_path,
        model=effective_model,
        draft_sql_model=effective_draft,
        max_turns=max_turns,
        verbose=verbose,
        hint=hint,
    )

