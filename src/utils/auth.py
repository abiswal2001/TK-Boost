import os

# Minimal provider toggle and credentials (kept consistent with original script)
USE_OPENAI = False  # set to True to use OpenAI instead of Azure OpenAI

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

# Azure OpenAI
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")


def configure_llm_env():
    """Configure environment variables for the selected LLM provider.

    Mirrors the original script's behavior by setting provider-specific env vars.
    """
    if USE_OPENAI:
        if OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        if OPENAI_API_BASE:
            os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
    else:
        os.environ["AZURE_API_KEY"] = AZURE_API_KEY
        os.environ["AZURE_API_BASE"] = AZURE_API_BASE
        os.environ["AZURE_API_VERSION"] = AZURE_API_VERSION




