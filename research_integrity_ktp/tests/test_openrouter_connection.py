#!/usr/bin/env python3
"""Quick test to verify OpenRouter API connection."""

import asyncio

from dotenv import load_dotenv

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.model._chat_message import ChatMessageUser

load_dotenv()


async def test_connection():
    """Test basic OpenRouter connection."""
    print("Testing OpenRouter connection...")

    # Try a simple, fast free model
    model = get_model(
        "openrouter/google/gemini-2.0-flash-exp:free",
        config=GenerateConfig(max_tokens=100),
    )

    print(f"Model: {model.name}")

    # Simple test message
    messages = [ChatMessageUser(content="Say 'Hello, OpenRouter!' and nothing else.")]

    print("Sending test message...")
    result = await model.generate(
        input=messages,
        tools=[],
        tool_choice="none",
        config=GenerateConfig(max_tokens=100),
    )

    print(f"Response: {result.completion}")
    print("✅ OpenRouter connection successful!")

    return result


if __name__ == "__main__":
    asyncio.run(test_connection())
