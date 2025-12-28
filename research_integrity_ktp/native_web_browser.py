#!/usr/bin/env python3
"""
Native web browser implementation using Playwright + structured outputs.

Reimplements Inspect AI's web_browser tools but using:
- NATIVE Playwright for all browser operations
- Structured outputs for LLM decisions
- Pydantic @model_validator for execution
- Accessibility tree extraction from Playwright

Architecture:
1. Launch Playwright browser
2. Navigate to URL, get accessibility tree
3. LLM sees tree + menu of actions
4. LLM outputs structured decision
5. Pydantic executes action with Playwright
6. Get new tree, back to step 3
7. Repeat until LLM outputs "done"

Usage:
    from native_web_browser import BrowserSession, browse_with_llm

    result = browse_with_llm(
        url="http://example.com",
        goal="Find h-index and citations",
        model=model,
    )
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Literal

from playwright.async_api import Page, async_playwright
from pydantic import BaseModel, Field, model_validator

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.solver import generate
from inspect_ai.util import json_schema


# ============================================================================
# Accessibility tree extraction from Playwright
# ============================================================================


async def get_accessibility_tree(page: Page) -> str:
    """
    Extract accessibility tree from Playwright page.
    Returns a text representation similar to Inspect AI's web_browser tool.
    """
    # Get accessibility snapshot from Playwright
    snapshot = await page.accessibility.snapshot()

    if not snapshot:
        return "[No accessibility tree available]"

    # Convert to text format with element IDs
    lines = []
    element_id_counter = [1]  # Use list to allow mutation in nested function

    def format_node(node, depth=0):
        indent = "  " * depth
        node_id = element_id_counter[0]
        element_id_counter[0] += 1

        # Extract node info
        role = node.get("role", "")
        name = node.get("name", "")

        # Format line
        if name:
            line = f"{indent}[{node_id}] {role} \"{name}\""
        else:
            line = f"{indent}[{node_id}] {role}"

        # Add additional properties
        props = []
        if node.get("focused"):
            props.append("focused: True")
        if node.get("value"):
            props.append(f"value: {node.get('value')}")
        if node.get("description"):
            props.append(f"description: {node.get('description')}")

        if props:
            line += " [" + ", ".join(props) + "]"

        lines.append(line)

        # Process children
        children = node.get("children", [])
        for child in children:
            format_node(child, depth + 1)

    format_node(snapshot)
    return "\n".join(lines)


async def find_element_by_id(page: Page, element_id: int) -> str | None:
    """
    Find element selector by ID from accessibility tree.

    This is a simplified version - in production you'd maintain
    a mapping from element_id to actual Playwright locators.
    """
    # For now, return a generic selector
    # In production, you'd build this mapping during tree extraction
    return f"[aria-label], [role], text={element_id}"


# ============================================================================
# Browser action models with structured outputs
# ============================================================================


class BrowserAction_Go(BaseModel):
    """LLM decides to navigate to a URL."""

    action: Literal["go"] = "go"
    url: str = Field(..., description="URL to navigate to")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Click(BaseModel):
    """LLM decides to click an element."""

    action: Literal["click"] = "click"
    element_id: int = Field(..., description="ID of element to click")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Type(BaseModel):
    """LLM decides to type into an element."""

    action: Literal["type"] = "type"
    element_id: int = Field(..., description="ID of element to type into")
    text: str = Field(..., description="Text to type")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_TypeSubmit(BaseModel):
    """LLM decides to type and press ENTER."""

    action: Literal["type_submit"] = "type_submit"
    element_id: int = Field(..., description="ID of element to type into")
    text: str = Field(..., description="Text to type before ENTER")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Scroll(BaseModel):
    """LLM decides to scroll."""

    action: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"] = Field(..., description="Scroll direction")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Back(BaseModel):
    """LLM decides to go back."""

    action: Literal["back"] = "back"

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Done(BaseModel):
    """LLM decides task is complete."""

    action: Literal["done"] = "done"
    result: str = Field(..., description="Final result/answer")


# Union type for all actions
BrowserAction = (
    BrowserAction_Go
    | BrowserAction_Click
    | BrowserAction_Type
    | BrowserAction_TypeSubmit
    | BrowserAction_Scroll
    | BrowserAction_Back
    | BrowserAction_Done
)


# ============================================================================
# Browser session manager
# ============================================================================


class BrowserSession:
    """
    Manages a Playwright browser session.
    Executes browser actions and extracts accessibility trees.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.browser = None
        self.context = None
        self.page = None
        self.action_counter = 0

    async def start(self):
        """Launch browser."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        self.context = await self.browser.new_context(
            ignore_https_errors=True,
            bypass_csp=True,
        )
        self.page = await self.context.new_page()

    async def execute_action(self, action: BrowserAction) -> str:
        """
        Execute browser action and return new accessibility tree.
        """
        self.action_counter += 1

        if action.action == "go":
            await self.page.goto(action.url, timeout=30000, wait_until='domcontentloaded')

        elif action.action == "click":
            # Find element and click
            # Simplified: use element_id as part of selector
            # In production, maintain proper element_id -> locator mapping
            await self.page.click(f"nth=0")  # Placeholder

        elif action.action == "type":
            await self.page.fill(f"nth=0", action.text)  # Placeholder

        elif action.action == "type_submit":
            await self.page.fill(f"nth=0", action.text)  # Placeholder
            await self.page.press(f"nth=0", "Enter")

        elif action.action == "scroll":
            if action.direction == "down":
                await self.page.evaluate("window.scrollBy(0, window.innerHeight)")
            else:
                await self.page.evaluate("window.scrollBy(0, -window.innerHeight)")

        elif action.action == "back":
            await self.page.go_back()

        # Wait for page to settle
        try:
            await self.page.wait_for_load_state('networkidle', timeout=5000)
        except:
            pass

        # Get new accessibility tree
        tree = await get_accessibility_tree(self.page)

        # Take screenshot
        screenshot_path = self.storage_dir / f"action_{self.action_counter}_screenshot.png"
        await self.page.screenshot(path=str(screenshot_path))

        # Save HTML dump
        html_path = self.storage_dir / f"action_{self.action_counter}_page.html"
        html_content = await self.page.content()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Store in action object
        if hasattr(action, 'accessibility_tree_'):
            action.accessibility_tree_ = tree
            action.screenshot_path_ = screenshot_path

        return tree

    async def close(self):
        """Close browser."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


# ============================================================================
# Main browsing function with LLM loop
# ============================================================================


def browse_with_llm(
    url: str,
    goal: str,
    model,
    max_actions: int = 20,
    storage_dir: Path = Path("logs/native_browser"),
) -> dict:
    """
    Browse a URL using LLM + Playwright.

    Args:
        url: Initial URL to visit
        goal: What to find/accomplish
        model: Inspect AI model
        max_actions: Maximum browser actions
        storage_dir: Where to save screenshots/HTML

    Returns:
        dict with result and action history
    """
    print(f"\n[Native Browser] Starting session...")
    print(f"  URL: {url}")
    print(f"  Goal: {goal}")

    # Start browser session
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    session = BrowserSession(storage_dir)
    loop.run_until_complete(session.start())

    try:
        # Initial navigation
        initial_action = BrowserAction_Go(url=url)
        tree = loop.run_until_complete(session.execute_action(initial_action))

        action_history = []
        action_count = 0

        while action_count < max_actions:
            action_count += 1

            print(f"\n  [Action {action_count}] Asking LLM for next action...")

            # Build prompt with current tree + available actions
            prompt = f"""You are browsing a web page to: {goal}

Current page accessibility tree:
{tree[:2000]}  # Truncate for context

Available actions:
- go: Navigate to a URL
- click: Click an element by ID
- type: Type text into an element
- type_submit: Type and press ENTER
- scroll: Scroll up or down
- back: Go back
- done: Task complete, provide result

Choose your next action."""

            # Get LLM decision
            @task
            def browser_decision_task():
                return Task(
                    dataset=[Sample(input=prompt, target="")],
                    solver=generate(),
                    config=GenerateConfig(
                        response_schema=ResponseSchema(
                            name="BrowserAction",
                            json_schema={
                                "oneOf": [
                                    json_schema(BrowserAction_Go),
                                    json_schema(BrowserAction_Click),
                                    json_schema(BrowserAction_Type),
                                    json_schema(BrowserAction_TypeSubmit),
                                    json_schema(BrowserAction_Scroll),
                                    json_schema(BrowserAction_Back),
                                    json_schema(BrowserAction_Done),
                                ]
                            },
                            strict=True,
                        ),
                        max_tokens=512,
                    ),
                )

            log = eval(browser_decision_task(), model=model)[0]
            decision_json = json.loads(log.samples[0].output.completion)

            # Parse action
            action_type = decision_json.get("action")

            if action_type == "done":
                action = BrowserAction_Done(**decision_json)
                print(f"  ✓ Task complete: {action.result}")
                action_history.append(action.model_dump())
                break

            # Create appropriate action object
            if action_type == "go":
                action = BrowserAction_Go(**decision_json)
            elif action_type == "click":
                action = BrowserAction_Click(**decision_json)
            elif action_type == "type":
                action = BrowserAction_Type(**decision_json)
            elif action_type == "type_submit":
                action = BrowserAction_TypeSubmit(**decision_json)
            elif action_type == "scroll":
                action = BrowserAction_Scroll(**decision_json)
            elif action_type == "back":
                action = BrowserAction_Back(**decision_json)
            else:
                print(f"  ✗ Unknown action: {action_type}")
                break

            print(f"  ✓ Action: {action.action}")

            # Execute action
            tree = loop.run_until_complete(session.execute_action(action))
            action_history.append(action.model_dump())

        # Save history
        history_file = storage_dir / "action_history.json"
        with open(history_file, "w") as f:
            json.dump(action_history, f, indent=2)

        print(f"\n  ✓ Session complete, {len(action_history)} actions")
        print(f"  ✓ History saved: {history_file}")

        return {
            "action_history": action_history,
            "actions_taken": len(action_history),
            "final_result": action_history[-1].get("result", "") if action_history else "",
        }

    finally:
        loop.run_until_complete(session.close())
        loop.close()


# ============================================================================
# Test function
# ============================================================================


if __name__ == "__main__":
    # Test with example.com
    model = get_model("openrouter/qwen/qwen3-coder:free")

    result = browse_with_llm(
        url="http://example.com",
        goal="Read the page title and first paragraph",
        model=model,
        max_actions=5,
    )

    print("\n" + "=" * 70)
    print("BROWSING COMPLETE")
    print("=" * 70)
    print(f"\nActions taken: {result['actions_taken']}")
    print(f"Final result: {result['final_result']}")
