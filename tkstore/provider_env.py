import os


def _get_env(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    return val if val is not None else default


def _is_azure_model(model: str) -> bool:
    return str(model or "").lower().startswith("azure/")


def _detect_provider(model: str) -> str:
    """Detect provider at runtime from model/env.

    Priority:
      1) model prefix (azure/* => azure)
      2) OPENAI_API_KEY => openai
      3) AZURE_API_KEY / AZURE_OPENAI_API_KEY => azure
      4) fallback openai
    """
    if _is_azure_model(model):
        return "azure"
    if _get_env("OPENAI_API_KEY"):
        return "openai"
    if _get_env("AZURE_API_KEY") or _get_env("AZURE_OPENAI_API_KEY"):
        return "azure"
    return "openai"


def provider_info(model: str) -> str:
    """Return a short string describing which provider/model will be used."""
    provider = _detect_provider(model)
    if provider == "openai":
        openai_base = _get_env("OPENAI_API_BASE", "https://api.openai.com/v1")
        return f"OpenAI (base={openai_base}) model={model}"

    azure_base = _get_env("AZURE_API_BASE") or _get_env("AZURE_OPENAI_ENDPOINT")
    azure_version = _get_env("AZURE_API_VERSION", "2024-12-01-preview")
    return f"Azure OpenAI (endpoint={azure_base} version={azure_version}) model={model}"


def log_provider(model: str):
    print(f"[provider] using: {provider_info(model)}")


