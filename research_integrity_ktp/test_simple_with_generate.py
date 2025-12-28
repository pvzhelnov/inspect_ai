#!/usr/bin/env python3
"""Test with generate() solver to isolate the issue."""

from dotenv import load_dotenv

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.solver import generate

load_dotenv()


@task
def task_with_generate():
    return Task(
        dataset=[Sample(input="Say hello in 3 words", target="hello world there")],
        solver=generate(),
    )


print("Testing with generate() solver...")
model = get_model("openrouter/qwen/qwen3-4b:free")
print(f"Model: {model.name}")

print("Running eval...")
result = eval(task_with_generate(), model=model)
print(f"Success! Result: {result}")
