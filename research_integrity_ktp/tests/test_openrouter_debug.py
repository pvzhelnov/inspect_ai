#!/usr/bin/env python3
"""Debug OpenRouter model name handling."""

import asyncio

from dotenv import load_dotenv

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.model._chat_message import ChatMessageUser

load_dotenv()


async def test():
    print("Testing OpenRouter model name handling...")

    # Get the model
    model = get_model("openrouter/qwen/qwen3-4b:free")

    # Check the model name and service_model_name
    print(f"Full model_name: {model.name}")
    if hasattr(model, "api"):
        api = model.api
        print(f"API model_name: {api.model_name}")
        if hasattr(api, "service_model_name"):
            print(f"Service model_name: {api.service_model_name()}")
        if hasattr(api, "service"):
            print(f"Service: {api.service}")

    # Try a simple generation
    print("\nAttempting generation...")
    messages = [ChatMessageUser(content="Say hello in 5 words")]

    try:
        output, call = await model.api.generate(
            input=messages,
            tools=[],
            tool_choice="none",
            config=GenerateConfig(max_tokens=4096),
        )
        msg = output.choices[0].message.content
        reasoning = next(c.reasoning for c in msg if c.type == "reasoning")
        response = next(c.text for c in msg if c.type == "text")
        print(f"Success! Response: {response}")
        print(f"Reasoning: {reasoning}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())
