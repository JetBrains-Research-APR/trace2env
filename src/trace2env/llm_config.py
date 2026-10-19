"""Shared LiteLLM configuration for every network-backed model client."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, find_dotenv


def _settings() -> dict[str, str | None]:
    # Anchor editable checkouts to their own .env even when a script changes cwd.
    # Installed wheels instead find the project's .env from the working directory.
    root = Path(__file__).resolve().parents[2]
    path = root / ".env" if (root / "pyproject.toml").is_file() else find_dotenv(usecwd=True)
    values = dotenv_values(path) if path else {}
    return {name: os.environ.get(name, values.get(name))
            for name in ("LITELLM_API_BASE", "LITELLM_API_KEY")}


def litellm_base_url() -> str | None:
    """Endpoint for run manifests; reading it does not require credentials."""
    return _settings()["LITELLM_API_BASE"]


def _credentials() -> tuple[str, str]:
    values = _settings()
    missing = [name for name, value in values.items() if not value or not value.strip()]
    if missing:
        raise RuntimeError("Set " + " and ".join(missing) + " in the project .env or environment to use LiteLLM")
    return values["LITELLM_API_BASE"].strip(), values["LITELLM_API_KEY"].strip()


def create_llm_client(*, timeout: float = 1800.0, max_retries: int = 0) -> Any:
    """Build an OpenAI-compatible client targeting only the configured LiteLLM gateway."""
    from openai import OpenAI

    base_url, api_key = _credentials()
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


def main() -> None:
    """Run external benchmark clients with the same endpoint and key, without shelling out secrets."""
    if sys.argv[1:] == ["base-url"]:
        base_url, _ = _credentials()
        print(base_url)
        return
    if len(sys.argv) < 3 or sys.argv[1] != "run":
        raise SystemExit("Usage: python -m trace2env.llm_config base-url | run COMMAND [ARGS...]")
    base_url, api_key = _credentials()
    environment = dict(os.environ, LITELLM_API_BASE=base_url, LITELLM_API_KEY=api_key,
                       OPENAI_BASE_URL=base_url, OPENAI_API_BASE=base_url, OPENAI_API_KEY=api_key)
    os.execvpe(sys.argv[2], sys.argv[2:], environment)


if __name__ == "__main__":
    main()
