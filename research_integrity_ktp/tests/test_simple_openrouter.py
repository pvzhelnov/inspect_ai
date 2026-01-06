#!/usr/bin/env python3
"""Minimal OpenRouter test with detailed debugging."""

import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("STEP 1: Import inspect_ai modules")
print("=" * 70)

try:
    from inspect_ai import Task, eval, task

    print("✓ Imported Task, eval, task")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from inspect_ai.dataset import Sample

    print("✓ Imported Sample")
except Exception as e:
    print(f"✗ Failed to import Sample: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from inspect_ai.model import get_model

    print("✓ Imported GenerateConfig, get_model")
except Exception as e:
    print(f"✗ Failed to import model: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 2: Get OpenRouter model")
print("=" * 70)

try:
    model = get_model("openrouter/qwen/qwen3-coder:free")
    print(f"✓ Got model: {model.name}")
except Exception as e:
    print(f"✗ Failed to get model: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 3: Create simple task")
print("=" * 70)

try:

    @task
    def simple_task():
        return Task(
            dataset=[Sample(input="Say hello", target="hello")],
            solver=[],
        )

    print("✓ Created task")
except Exception as e:
    print(f"✗ Failed to create task: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 4: Run eval (this is where it might hang)")
print("=" * 70)

try:
    print("About to call eval()...")
    result = eval(
        simple_task(),
        model=model,
    )
    print("✓ Eval completed successfully")
    print(f"   Results: {result}")
except Exception as e:
    print(f"✗ Eval failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 70)
