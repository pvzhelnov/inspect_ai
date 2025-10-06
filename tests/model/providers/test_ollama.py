from pydantic import BaseModel

from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.model._providers.ollama import OllamaAPI
from inspect_ai.util import json_schema


class Color(BaseModel):
    red: int
    green: int
    blue: int


def test_ollama_structured_output_completion_params() -> None:
    api = OllamaAPI.__new__(OllamaAPI)
    api.model_name = "ollama/llama3.1"
    api.service = "Ollama"

    config = GenerateConfig(
        response_schema=ResponseSchema(
            name="color",
            description="RGB color values",
            json_schema=json_schema(Color),
            strict=True,
        )
    )

    params = OllamaAPI.completion_params(api, config=config, tools=False)

    assert "response_format" not in params
    assert params["format"] == {
        "type": "object",
        "properties": {
            "red": {"type": "integer"},
            "green": {"type": "integer"},
            "blue": {"type": "integer"},
        },
        "additionalProperties": False,
        "required": ["red", "green", "blue"],
        "description": "RGB color values",
        "title": "color",
    }

