import os

# --- Provider/Env (mirrors refiner/runner for self-sufficiency) ---
USE_OPENAI = False
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")

if USE_OPENAI:
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if OPENAI_API_BASE:
        os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
else:
    os.environ["AZURE_API_KEY"] = AZURE_API_KEY
    os.environ["AZURE_API_BASE"] = AZURE_API_BASE
    os.environ["AZURE_API_VERSION"] = AZURE_API_VERSION


def provider_info(model: str) -> str:
    """Return a short string describing which provider and model will be used."""
    if USE_OPENAI:
        return f"OpenAI (base={OPENAI_API_BASE}) model={model}"
    return f"Azure OpenAI (endpoint={AZURE_API_BASE} version={AZURE_API_VERSION}) model={model}"


def log_provider(model: str):
    info = provider_info(model)
    print(f"[provider] using: {info}")


