from __future__ import annotations

import os
from typing import Any, cast

import httpx
from openai import (
    APIStatusError,
    BadRequestError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion
from typing_extensions import override

from inspect_ai.model._openai import (
    chat_choices_from_openai,
    messages_to_openai,
    model_output_from_openai,
    openai_chat_tool_choice,
    openai_chat_tools,
    openai_completion_params,
    openai_handle_bad_request,
    openai_media_filter,
    openai_should_retry,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from .._chat_message import ChatMessage
from .._generate_config import GenerateConfig
from .._model import ModelAPI
from .._model_call import ModelCall
from .._model_output import ChatCompletionChoice, ModelOutput
from .openai_compatible import OpenAICompatibleHandler
from .openai_compatible import _resolve_chat_choice as resolve_chat_choice
from .util import model_base_url
from .util.chatapi import chat_api_messages_for_handler
from .util.hooks import HttpxHooks


def _strip_not_given(value: Any) -> Any:
    """Recursively remove OpenAI NOT_GIVEN sentinels from request payloads."""
    if isinstance(value, dict):
        return {
            key: _strip_not_given(item)
            for key, item in value.items()
            if item is not NOT_GIVEN
        }
    if isinstance(value, list):
        return [_strip_not_given(item) for item in value if item is not NOT_GIVEN]
    if isinstance(value, tuple):
        return tuple(_strip_not_given(item) for item in value if item is not NOT_GIVEN)
    return value


def _error_message(response: httpx.Response, body: Any) -> str:
    """Extract a human-readable error message from an Ollama response."""
    if isinstance(body, dict):
        error_field = body.get("error")
        if isinstance(error_field, dict):
            message = error_field.get("message")
            if isinstance(message, str):
                return message
        if isinstance(error_field, str):
            return error_field
        message_field = body.get("message")
        if isinstance(message_field, str):
            return message_field

    text = response.text
    if text:
        return text

    return f"Ollama request failed with status {response.status_code}"


class OllamaAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        emulate_tools: bool = False,
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=["OLLAMA_API_KEY"],
            config=config,
        )

        self.service = "Ollama"
        if not self.api_key:
            self.api_key = os.environ.get("OLLAMA_API_KEY", None) or "ollama"

        resolved_base = model_base_url(base_url, "OLLAMA_BASE_URL")
        if not resolved_base:
            resolved_base = "http://localhost:11434"

        resolved_base = resolved_base.rstrip("/")
        if resolved_base.endswith("/api/chat"):
            self.base_url = resolved_base[: -len("/api/chat")]
            self._endpoint = resolved_base
        elif resolved_base.endswith("/api"):
            self.base_url = resolved_base[:-4]
            self._endpoint = f"{resolved_base}/chat"
        else:
            self.base_url = resolved_base
            self._endpoint = f"{resolved_base}/api/chat"

        http_client = model_args.pop("http_client", None)
        if http_client is None:
            from inspect_ai.model._openai import OpenAIAsyncHttpxClient

            http_client = OpenAIAsyncHttpxClient()

        self.client = http_client
        self.emulate_tools = emulate_tools

        default_headers: dict[str, str] = {}
        if self.api_key:
            default_headers["Authorization"] = f"Bearer {self.api_key}"
        self._default_headers = default_headers

        self._http_hooks = HttpxHooks(self.client)

    @override
    async def aclose(self) -> None:
        await self.client.aclose()

    @override
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        tools, tool_choice, config = self.resolve_tools(tools, tool_choice, config)

        handler: OpenAICompatibleHandler | None
        if self.emulate_tools and tools:
            handler = OpenAICompatibleHandler(self.model_name)
        else:
            handler = None

        if handler is not None:
            input = chat_api_messages_for_handler(input, tools, handler)

        request_id = self._http_hooks.start_request()
        request: dict[str, Any] = {}
        response: dict[str, Any] = {}

        def model_call() -> ModelCall:
            return ModelCall.create(
                request=request,
                response=response,
                filter=openai_media_filter,
                time=self._http_hooks.end_request(request_id),
            )

        have_tools = bool(tools) and not self.emulate_tools
        completion_params = self.completion_params(config=config, tools=have_tools)

        request = dict(
            messages=await messages_to_openai(input),
            tools=openai_chat_tools(tools) if have_tools else NOT_GIVEN,
            tool_choice=openai_chat_tool_choice(tool_choice)
            if have_tools
            else NOT_GIVEN,
            extra_headers={HttpxHooks.REQUEST_ID_HEADER: request_id},
            stream=False,
            **completion_params,
        )

        try:
            completion = await self._generate_completion(request, config)
            response = completion.model_dump()
            self.on_response(response)
            choices = self.chat_choices_from_completion(completion, tools)

            if handler is not None:
                choices = [
                    resolve_chat_choice(choice, tools, handler) for choice in choices
                ]

            output = model_output_from_openai(completion, choices)
            return output, model_call()
        except (
            BadRequestError,
            UnprocessableEntityError,
            PermissionDeniedError,
        ) as ex:
            return self.handle_bad_request(ex), model_call()

    def resolve_tools(
        self, tools: list[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig
    ) -> tuple[list[ToolInfo], ToolChoice, GenerateConfig]:
        return tools, tool_choice, config

    async def _generate_completion(
        self, request: dict[str, Any], config: GenerateConfig
    ) -> ChatCompletion:
        api_request = dict(request)

        extra_headers = api_request.pop("extra_headers", NOT_GIVEN)
        headers: dict[str, str] = dict(self._default_headers)
        if isinstance(extra_headers, dict):
            headers.update({k: str(v) for k, v in extra_headers.items()})
        elif extra_headers not in (NOT_GIVEN, None):
            headers.update(dict(cast(dict[str, str], extra_headers)))

        extra_query = api_request.pop("extra_query", NOT_GIVEN)
        query_params: dict[str, Any] | None
        if isinstance(extra_query, dict):
            query_params = {
                key: value
                for key, value in extra_query.items()
                if value is not NOT_GIVEN
            }
        else:
            query_params = None

        api_request.pop("extra_timeout", None)
        extra_body = api_request.pop("extra_body", NOT_GIVEN)

        payload = {
            key: _strip_not_given(value)
            for key, value in api_request.items()
            if value is not NOT_GIVEN
        }

        if extra_body not in (NOT_GIVEN, None):
            body_fragment = (
                dict(extra_body)
                if isinstance(extra_body, dict)
                else _strip_not_given(extra_body)
            )
            if isinstance(body_fragment, dict):
                payload.update(_strip_not_given(body_fragment))

        response = await self.client.post(
            self._endpoint,
            json=payload,
            headers=headers,
            params=query_params,
        )

        if response.status_code >= 400:
            try:
                body: Any = response.json()
            except ValueError:
                body = None

            message = _error_message(response, body)

            if response.status_code == 400:
                raise BadRequestError(message, response=response, body=body)
            if response.status_code == 403:
                raise PermissionDeniedError(message, response=response, body=body)
            if response.status_code == 422:
                raise UnprocessableEntityError(message, response=response, body=body)

            raise APIStatusError(message, response=response, body=body)

        data = response.json()
        return ChatCompletion.model_validate(data)

    def completion_params(self, config: GenerateConfig, tools: bool) -> dict[str, Any]:
        params = openai_completion_params(
            model=self.service_model_name(),
            config=config,
            tools=tools,
        )

        if config.response_schema is not None:
            schema = config.response_schema.json_schema.model_dump(exclude_none=True)
            params.pop("response_format", None)
            params["format"] = schema

        effort = params.pop("reasoning_effort", None)
        if effort is not None:
            params.setdefault("extra_body", {})["reasoning"] = {"effort": effort}

        return params

    def chat_choices_from_completion(
        self, completion: ChatCompletion, tools: list[ToolInfo]
    ) -> list[ChatCompletionChoice]:
        return chat_choices_from_openai(completion, tools)

    def handle_bad_request(self, ex: APIStatusError) -> ModelOutput | Exception:
        return openai_handle_bad_request(self.service_model_name(), ex)

    def service_model_name(self) -> str:
        if self.model_name.lower().startswith("ollama/"):
            return self.model_name.split("/", 1)[1]
        return self.model_name

    @override
    def should_retry(self, ex: BaseException) -> bool:  # type: ignore[override]
        return openai_should_retry(ex)

    @override
    def connection_key(self) -> str:
        return str(self.api_key)

    def on_response(self, response: dict[str, Any]) -> None:  # pragma: no cover
        pass
