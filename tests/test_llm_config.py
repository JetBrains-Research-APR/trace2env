"""Check routing and authentication at the HTTP boundary without making network calls."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from trace2env import llm_config
from trace2env.llm import OpenAIChatLLM, OpenAIChatStructuredLLM, OpenAIResponsesLLM, OpenAIToolLLM
from trace2env.models import Confidence


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    path = tmp_path / ".env"
    path.write_text('LITELLM_API_BASE="https://gateway.example/v1"\nLITELLM_API_KEY="internal-key"\n')
    monkeypatch.setattr(llm_config, "__file__", str(tmp_path / "src" / "trace2env" / "llm_config.py"))
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://external.example/v1")
    monkeypatch.chdir(tmp_path.parent)
    return path


@pytest.mark.parametrize("adapter", [OpenAIResponsesLLM, OpenAIChatStructuredLLM, OpenAIChatLLM, OpenAIToolLLM])
def test_all_transports_use_dotenv_gateway_and_key(configuration, adapter):
    requests = []
    content = '{"value": 0.5, "rationale": "OK"}'

    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/responses"):
            body = {"id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
                    "model": "test-model", "output": [{"id": "msg_1", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text", "text": content, "annotations": []}]}]}
        else:
            body = {"id": "chatcmpl_1", "object": "chat.completion", "created": 0, "model": "test-model",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}
        return httpx.Response(200, json=body)

    def client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    with patch("openai.OpenAI", side_effect=client):
        llm = adapter(model="test-model")
    try:
        if adapter in (OpenAIResponsesLLM, OpenAIChatStructuredLLM):
            assert llm.complete(system="Return JSON", user="OK", response_model=Confidence, role="test").value == 0.5
        elif adapter is OpenAIChatLLM:
            assert llm.chat(messages=[{"role": "user", "content": "OK"}], role="test") == content
        else:
            assert llm.respond(messages=[{"role": "user", "content": "OK"}], tools=[], role="test").content == content
    finally:
        llm.client.close()
    assert len(requests) == 1
    request = requests[0]
    assert request.url.host == "gateway.example"
    assert request.url.path == ("/v1/responses" if adapter is OpenAIResponsesLLM else "/v1/chat/completions")
    assert request.headers["Authorization"] == "Bearer internal-key"
    assert json.loads(request.content)["model"] == "test-model"
    assert json.loads(request.content)["temperature"] == 1.0


def test_environment_overrides_dotenv_and_preserves_client_limits(configuration, monkeypatch):
    monkeypatch.setenv("LITELLM_API_BASE", "https://deployment.example/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "deployment-key")
    with llm_config.create_llm_client(timeout=180, max_retries=2) as client:
        assert str(client.base_url) == "https://deployment.example/v1/"
        assert client.api_key == "deployment-key"
        assert client.timeout == 180
        assert client.max_retries == 2


@pytest.mark.parametrize("contents,missing", [
    ("", "LITELLM_API_BASE and LITELLM_API_KEY"),
    ("LITELLM_API_BASE=https://gateway.example/v1\n", "LITELLM_API_KEY"),
    ("LITELLM_API_KEY=internal-key\n", "LITELLM_API_BASE"),
])
def test_missing_config_fails_without_external_fallback(configuration, contents, missing):
    configuration.write_text(contents)
    with pytest.raises(RuntimeError, match=missing):
        llm_config.create_llm_client()
    # Offline and scripted providers can still inject a client without credentials.
    fake = object()
    assert OpenAIResponsesLLM(client=fake).client is fake


def test_external_runner_exports_credentials_without_command_line_secrets(configuration, monkeypatch):
    monkeypatch.setattr("sys.argv", ["llm_config", "run", "benchmark", "--model", "test-model"])
    with patch("os.execvpe") as execute:
        llm_config.main()
    command, arguments, environment = execute.call_args.args
    assert command == "benchmark"
    assert arguments == ["benchmark", "--model", "test-model"]
    assert environment["OPENAI_API_KEY"] == environment["LITELLM_API_KEY"] == "internal-key"
    assert environment["OPENAI_BASE_URL"] == environment["OPENAI_API_BASE"] == "https://gateway.example/v1"
    assert os.environ["OPENAI_API_KEY"] == "unrelated-key"


def test_installed_package_finds_project_dotenv(configuration, tmp_path, monkeypatch):
    monkeypatch.setattr(llm_config, "__file__", str(tmp_path / "installed" / "lib" / "trace2env" / "llm_config.py"))
    child = tmp_path / "experiment"
    child.mkdir()
    monkeypatch.chdir(child)
    with llm_config.create_llm_client() as client:
        assert client.api_key == "internal-key"
        assert str(client.base_url) == "https://gateway.example/v1/"


def test_gateway_is_part_of_cache_identity(configuration, monkeypatch):
    from trace2env.llm import _provider_identity

    first = OpenAIChatLLM(model="test-model")
    monkeypatch.setenv("LITELLM_API_BASE", "https://another-gateway.example/v1")
    second = OpenAIChatLLM(model="test-model")
    try:
        assert _provider_identity(first)["base_url"] == "https://gateway.example/v1/"
        assert _provider_identity(first) != _provider_identity(second)
        assert "internal-key" not in json.dumps(_provider_identity(first))
    finally:
        first.client.close()
        second.client.close()


def test_sampling_retains_supported_temperatures():
    from trace2env.llm import _sampling

    assert "temperature" not in _sampling(0.6, 100, None, model="openai/gpt-5.6-sol")
    assert _sampling(1, 100, None, model="openai/gpt-5.6-sol")["temperature"] == 1
    assert _sampling(0.6, 100, "none", model="openai/gpt-5.6-sol")["temperature"] == 0.6
    assert _sampling(0.6, 100, None, model="openai/gpt-5.2")["temperature"] == 0.6


def test_cli_defaults_keep_judge_sampling_separate():
    import argparse
    from trace2env.cli import _add_model_options, build_parser
    from trace2env.models import ReconstructionConfig

    parser = argparse.ArgumentParser()
    _add_model_options(parser)
    args = parser.parse_args([])
    assert args.model == "openai/gpt-6.1-sol"
    assert args.structured_temperature == 1.0
    assert ReconstructionConfig(environment_id="test", name="Test").model == args.model
    assert OpenAIResponsesLLM(client=object()).model == args.model

    parser = build_parser()
    prompting = parser.parse_args(["awb-run", "rows.jsonl", "--mode", "prompting", "--output", "pred.jsonl"])
    assert prompting.model == args.model
    assert prompting.temperature == 1.0
    long_horizon = parser.parse_args(["envscaler-long-horizon", "package", "--env-defs", "defs",
                                     "--env-id", "test", "--task-id", "task", "--output", "out"])
    assert long_horizon.task_temperature == 1.0
    judge = parser.parse_args(["awb-judge", "--predictions", "pred.jsonl", "--output", "judged.jsonl"])
    assert judge.judge_model == "openai/gpt-5.2-2025-12-11"
    assert judge.temperature == 0.6
