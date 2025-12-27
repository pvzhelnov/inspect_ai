#!/usr/bin/env python3
"""Debug OpenRouter model name handling."""

import asyncio
from inspect_ai.model import get_model, GenerateConfig
from inspect_ai.model._chat_message import ChatMessageUser

async def test():
    print("Testing OpenRouter model name handling...")

    # Get the model
    model = get_model("openrouter/qwen/qwen3-4b:free")

    # Check the model name and service_model_name
    print(f"Full model_name: {model.name}")
    if hasattr(model, 'api'):
        api = model.api
        print(f"API model_name: {api.model_name}")
        if hasattr(api, 'service_model_name'):
            print(f"Service model_name: {api.service_model_name()}")
        if hasattr(api, 'service'):
            print(f"Service: {api.service}")

    # Try a simple generation
    print("\nAttempting generation...")
    messages = [ChatMessageUser(content="Say hello in 5 words")]

    try:
        result = await model.api.generate(
            input=messages,
            tools=[],
            tool_choice="none",
            config=GenerateConfig(max_tokens=50),
        )
        print(f"Success! Response: {result.completion}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
