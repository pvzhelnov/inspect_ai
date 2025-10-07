from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict

from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.model._openai import model_output_from_openai
from inspect_ai.model._providers.ollama import OllamaAPI
from inspect_ai.util import json_schema


class Color(BaseModel):
    red: int
    green: int
    blue: int


class ReasoningStep(BaseModel):
    explanation: str
    output: str

    model_config = ConfigDict(extra="forbid")


class StructuredAnswer(BaseModel):
    steps: list[ReasoningStep]
    final_answer: str


@pytest.fixture()
def ollama_api() -> OllamaAPI:
    api = OllamaAPI.__new__(OllamaAPI)
    api.model_name = "ollama/llama3.1"
    api.service = "Ollama"
    return api


def test_ollama_structured_output_completion_params(ollama_api: OllamaAPI) -> None:
    config = GenerateConfig(
        response_schema=ResponseSchema(
            name="color",
            description="RGB color values",
            json_schema=json_schema(Color),
            strict=True,
        )
    )

    params = ollama_api.completion_params(config=config, tools=False)

    assert "response_format" not in params
    expected_schema = json_schema(Color).model_dump(exclude_none=True)
    assert params["format"] == expected_schema
    assert "extra_body" not in params or "format" not in params["extra_body"]


def test_ollama_structured_output_response_parsing(ollama_api: OllamaAPI) -> None:
    completion = ChatCompletion.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "ollama/qwen3:4b-fp16",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"red":255,"green":255,"blue":255}',
                    },
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 12,
                "total_tokens": 22,
            },
        }
    )

    choices = ollama_api.chat_choices_from_completion(completion, tools=[])
    output = model_output_from_openai(completion, choices)

    assert output.completion == '{"red":255,"green":255,"blue":255}'


@pytest.mark.asyncio()
async def test_ollama_generate_completion_uses_documented_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = ResponseSchema(
        name="structured_answer",
        json_schema=json_schema(StructuredAnswer),
    )
    config = GenerateConfig(
        response_schema=schema,
    )

    api = OllamaAPI(
        model_name="ollama/llama3.1",
        base_url="http://localhost:11434",
        api_key="ollama",
        config=config,
    )

    params = api.completion_params(config=config, tools=False)

    request: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "extra_headers": {"X-Test": "request"},
        "stream": False,
        **params,
    }

    captured: dict[str, Any] = {}

    async def fake_post(  # type: ignore[override]
        self: Any,
        path: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str] | None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        captured["path"] = path
        captured["json"] = json
        captured["headers"] = dict(headers or {})
        captured["params"] = params
        response_payload = {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "ollama/llama3.1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"steps":[{"explanation":"Greet the user","output":"Hello!"}],"final_answer":"Hello!"}',
                    },
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 24,
                "total_tokens": 36,
            },
        }
        return httpx.Response(
            status_code=200,
            json=response_payload,
            request=httpx.Request("POST", path),
        )

    monkeypatch.setattr(api.client.__class__, "post", fake_post, raising=False)

    completion = await api._generate_completion(request, config)

    assert captured["path"] == "http://localhost:11434/api/chat"
    assert captured["headers"]["X-Test"] == "request"
    assert captured["headers"]["Authorization"] == "Bearer ollama"
    assert captured["params"] is None
    expected_format = json_schema(StructuredAnswer).model_dump(exclude_none=True)
    assert captured["json"] == {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "stream": False,
        "format": expected_format,
    }
    assert (
        completion.choices[0].message.content
        == '{"steps":[{"explanation":"Greet the user","output":"Hello!"}],"final_answer":"Hello!"}'
    )

    await api.aclose()


@pytest.mark.asyncio()
async def test_ollama_generate_completion_merges_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GenerateConfig(reasoning_effort="medium")

    api = OllamaAPI(
        model_name="ollama/llama3.1",
        base_url="http://localhost:11434",
        api_key="ollama",
        config=config,
    )

    params = api.completion_params(config=config, tools=False)
    request: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Hi"}],
        "extra_headers": {"X-Test": "reasoning"},
        **params,
    }

    captured: dict[str, Any] = {}

    async def fake_post(
        self: Any,
        path: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str] | None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        captured["json"] = json
        captured["headers"] = dict(headers or {})
        captured["params"] = params
        return httpx.Response(
            status_code=200,
            json={
                "id": "cmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "ollama/llama3.1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hi"},
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            request=httpx.Request("POST", path),
        )

    monkeypatch.setattr(api.client.__class__, "post", fake_post, raising=False)

    await api._generate_completion(request, config)

    assert captured["headers"]["Authorization"] == "Bearer ollama"
    assert captured["headers"]["X-Test"] == "reasoning"
    assert captured["params"] is None
    assert captured["json"]["reasoning"] == {"effort": "medium"}
    assert "extra_body" not in captured["json"]

    await api.aclose()
