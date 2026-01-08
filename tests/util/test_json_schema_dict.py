"""Tests for JSONSchemaDict class."""
from typing import Union, Literal

import pytest
from pydantic import BaseModel, RootModel, Field

from inspect_ai.util._json import JSONSchema, JSONSchemaDict, json_schema
from inspect_ai.model import GenerateConfig, ResponseSchema


class SomeModelSimple(BaseModel):
    id: int
    field: str

class SomeModel1(BaseModel):
    id: Literal['model_1']
    field1: str

class SomeModel2(BaseModel):
    id: Literal['model_2']
    field2: str

class SomeModelAnyOf(RootModel[Union[SomeModel1,SomeModel2]]):
    """`JSONSchema` supports anyOf."""
    pass

class SomeModelOneOf(RootModel):
    """`JSONSchema` does not support oneOf as of the creation of these tests."""
    root: Union[SomeModel1,SomeModel2] = Field(..., discriminator='id')

def test_json_schema_dict_simple():
    """`JSONSchemaDict` should be equivalent to `JSONSchema` in cases where the latter fully supports the schema."""

    config_with_json_schema = GenerateConfig(
        response_schema=ResponseSchema(
            name="some",
            json_schema=json_schema(SomeModelSimple),
            strict=True,
        ),
    )

    config_with_json_schema_dict = GenerateConfig(
        response_schema=ResponseSchema(
            name="some",
            json_schema=JSONSchemaDict(SomeModelSimple.model_json_schema()),
            strict=True,
        ),
    )

    dump_json_schema = config_with_json_schema.model_dump()
    dump_json_schema_dict = config_with_json_schema_dict.model_dump()

    # There are still differences in the structure, so we only check property list
    assert dump_json_schema['response_schema']['json_schema']['properties'].keys() == dump_json_schema_dict['response_schema']['json_schema']['properties'].keys()
    # Ideally also add an assert on types and required, but may be unnecessary

def test_json_schema_dict_anyof():
    config_with_json_schema = GenerateConfig(
        response_schema=ResponseSchema(
            name="some",
            json_schema=json_schema(SomeModelAnyOf),
            strict=True,
        ),
    )

    config_with_json_schema_dict = GenerateConfig(
        response_schema=ResponseSchema(
            name="some",
            json_schema=JSONSchemaDict(SomeModelAnyOf.model_json_schema()),
            strict=True,
        ),
    )

    dump_json_schema = config_with_json_schema.model_dump()
    dump_json_schema_dict = config_with_json_schema_dict.model_dump()

    with pytest.xfail(reason="`JSONSchema` should support anyOf, so why does this fail?"):
        assert dump_json_schema == dump_json_schema_dict

def test_json_schema_dict_oneOf():
    config_with_json_schema = GenerateConfig(
        response_schema=ResponseSchema(
            name="some",
            json_schema=json_schema(SomeModelOneOf),
            strict=True,
        ),
    )

    config_with_json_schema_dict = GenerateConfig(
        response_schema=ResponseSchema(
            name="some",
            json_schema=JSONSchemaDict(SomeModelOneOf.model_json_schema()),
            strict=True,
        ),
    )

    dump_json_schema = config_with_json_schema.model_dump()
    dump_json_schema_dict = config_with_json_schema_dict.model_dump()

    assert dump_json_schema != dump_json_schema_dict
    print("As of the creation of this test, oneOf is unsupported by `JSONSchema` class.")
