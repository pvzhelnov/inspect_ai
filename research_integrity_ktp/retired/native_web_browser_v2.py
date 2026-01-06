#!/usr/bin/env python3
"""
Native web browser implementation using Playwright + structured outputs.
FIXED: All browser actions now go into ONE eval log instead of separate logs!

Uses proper Inspect AI pattern:
- ONE Task with ONE solver
- Solver calls generate() multiple times (one per action)
- All messages accumulate in state.messages
- state.store maintains browser session state
"""

import asyncio
import json
from pathlib import Path
from typing import Literal

from playwright.async_api import Page, async_playwright
from pydantic import BaseModel, Field

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, ResponseSchema
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver
from inspect_ai.util import json_schema

# Import from original file
from native_web_browser import (
    BrowserAction_Back,
    BrowserAction_Click,
    BrowserAction_Done,
    BrowserAction_Go,
    BrowserAction_Scroll,
    BrowserAction_Type,
    BrowserAction_TypeSubmit,
    BrowserSession,
    get_accessibility_tree,
)


# ============================================================================
# Browser state stored across multiple generate() calls
# ============================================================================


class BrowserSessionState(BaseModel):
    """State maintained across multiple generate() calls."""

    session: BrowserSession | None = Field(default=None, exclude=True)
    current_tree: str = ""
    action_history: list[dict] = Field(default_factory=list)
    action_count: int = 0
    max_actions: int = 20
    completed: bool = False
    final_result: str = ""
    goal: str = ""


# ============================================================================
# Solver that calls generate() multiple times - ALL IN ONE LOG!
# ============================================================================


@solver
def browser_agent_solver(
    url: str,
    goal: str,
    max_actions: int = 20,
    storage_dir: Path = Path("logs/native_browser"),
) -> Solver:
    """
    Browser agent that calls generate() multiple times.
    ALL actions go into ONE eval log!
    """

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        # Initialize browser session state in state.store
        browser_state = state.store.get("browser_state")
        if browser_state is None:
            browser_state = BrowserSessionState(
                max_actions=max_actions,
                goal=goal,
            )
            state.store.set("browser_state", browser_state)

            # Start Playwright browser
            loop = asyncio.get_event_loop()
            session = BrowserSession(storage_dir)
            await session.start()
            browser_state.session = session

            # Initial navigation
            initial_action = BrowserAction_Go(url=url)
            tree = await session.execute_action(initial_action)
            browser_state.current_tree = tree

            print(f"\n[Native Browser] Starting session...")
            print(f"  URL: {url}")
            print(f"  Goal: {goal[:100]}")

        # Browser action loop - each iteration calls generate()
        while (
            not browser_state.completed
            and browser_state.action_count < browser_state.max_actions
        ):
            browser_state.action_count += 1

            print(f"\n  [Action {browser_state.action_count}] Asking LLM for next action...")

            # Build action history summary
            history_summary = ""
            if browser_state.action_history:
                history_summary = "\n\nActions taken so far:\n"
                for i, prev_action in enumerate(browser_state.action_history[-5:], 1):
                    action_type = prev_action.get("action", "unknown")
                    if action_type == "go":
                        history_summary += f"{i}. Navigated to: {prev_action.get('url', 'N/A')}\n"
                    elif action_type == "click":
                        history_summary += f"{i}. Clicked element #{prev_action.get('element_id', 'N/A')}\n"
                    elif action_type == "type":
                        history_summary += f"{i}. Typed '{prev_action.get('text', 'N/A')}' into element #{prev_action.get('element_id', 'N/A')}\n"
                    elif action_type == "type_submit":
                        history_summary += f"{i}. Typed and submitted '{prev_action.get('text', 'N/A')}' in element #{prev_action.get('element_id', 'N/A')}\n"
                    elif action_type == "scroll":
                        history_summary += f"{i}. Scrolled {prev_action.get('direction', 'N/A')}\n"
                    elif action_type == "back":
                        history_summary += f"{i}. Went back\n"

            # Build prompt for this action
            prompt = f"""You are browsing a web page to: {browser_state.goal}
{history_summary}
Current page accessibility tree (ONLY VISIBLE ELEMENTS):
{browser_state.current_tree[:2000]}

Available actions:
- go: Navigate to a URL
- click: Click an element by ID
- type: Type text into an element
- type_submit: Type and press ENTER
- scroll: Scroll up or down
- back: Go back
- done: Task complete, provide result

Based on your previous actions and the current page, choose your next action."""

            # Add user message to conversation
            state.messages.append(ChatMessageUser(content=prompt))

            # *** KEY: Call generate() - appends to SAME state.messages! ***
            state = await generate_fn(
                state,
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

            # Parse LLM response
            completion = state.output.completion
            print(f"  DEBUG: LLM completion = {completion[:200]}")

            try:
                decision_json = json.loads(completion)
            except json.JSONDecodeError as e:
                print(f"  ✗ JSON decode error: {e}")
                browser_state.completed = True
                browser_state.final_result = f"Error: Invalid JSON - {e}"
                break

            # Handle invalid types
            if not isinstance(decision_json, (dict, list)):
                print(f"  ✗ LLM returned invalid type: {type(decision_json).__name__}")
                browser_state.completed = True
                browser_state.final_result = f"Error: Invalid type {type(decision_json).__name__}"
                break

            # Handle list outputs
            if isinstance(decision_json, list):
                if not decision_json:
                    print(f"  ✗ LLM returned empty list")
                    break
                decision_json = decision_json[0]

            # Flatten nested parameters
            if "parameters" in decision_json and isinstance(decision_json["parameters"], dict):
                params = decision_json.pop("parameters")
                decision_json.update(params)

            # Normalize field names
            action_type = decision_json.get("action") or decision_json.get("type")

            if "type" in decision_json and "action" not in decision_json:
                decision_json["action"] = decision_json.pop("type")

            if action_type == "type" and "query" in decision_json:
                decision_json["text"] = decision_json.pop("query")

            if action_type == "done" and "reason" in decision_json:
                decision_json["result"] = decision_json.pop("reason")

            if "element" in decision_json and "element_id" not in decision_json:
                element_val = decision_json.pop("element")
                try:
                    decision_json["element_id"] = int(element_val)
                except (ValueError, TypeError):
                    decision_json["element_id"] = element_val

            # Handle "done" action
            if action_type == "done":
                action = BrowserAction_Done(**decision_json)
                print(f"  ✓ Task complete: {action.result}")
                browser_state.action_history.append(action.model_dump())
                browser_state.completed = True
                browser_state.final_result = action.result
                break

            # Create action object
            try:
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
            except Exception as e:
                print(f"  ✗ Invalid action parameters: {e}")
                print(f"  ✗ Cannot complete task - no suitable elements found")
                action = BrowserAction_Done(
                    action="done",
                    result=f"Cannot complete: {str(e)}"
                )
                browser_state.action_history.append(action.model_dump())
                browser_state.completed = True
                browser_state.final_result = action.result
                break

            print(f"  ✓ Action: {action.action}")

            # Execute action with Playwright
            tree = await browser_state.session.execute_action(action)
            browser_state.current_tree = tree
            browser_state.action_history.append(action.model_dump())

        # Save action history
        history_file = storage_dir / "action_history.json"
        with open(history_file, "w") as f:
            json.dump(browser_state.action_history, f, indent=2)

        print(f"\n  ✓ Session complete, {len(browser_state.action_history)} actions")
        print(f"  ✓ History saved: {history_file}")

        # Close browser
        if browser_state.session:
            await browser_state.session.close()

        # Mark task complete
        state.completed = True

        return state

    return solve


# ============================================================================
# Main browsing function - creates ONE Task with the solver
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
    ALL browser actions go into ONE eval log!

    Args:
        url: Initial URL to visit
        goal: What to find/accomplish
        model: Inspect AI model
        max_actions: Maximum browser actions
        storage_dir: Where to save screenshots/HTML

    Returns:
        dict with result and action history
    """
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Create ONE Task with the browser agent solver
    @task
    def browser_task():
        return Task(
            dataset=[Sample(input=goal, target="")],
            solver=browser_agent_solver(
                url=url,
                goal=goal,
                max_actions=max_actions,
                storage_dir=storage_dir,
            ),
        )

    # *** KEY: eval() called ONCE for entire session! ***
    logs = eval(browser_task(), model=model)
    log = logs[0]

    # Extract results from the logged sample's store
    # Note: state.store is only accessible during solve(), so we get results from final_result in state
    sample = log.samples[0]

    # The solver stores action history in a file, load it from there
    history_file = storage_dir / "action_history.json"
    action_history = []
    final_result = ""

    if history_file.exists():
        with open(history_file, "r") as f:
            action_history = json.load(f)
        if action_history:
            # Get final result from last action if it was "done"
            last_action = action_history[-1]
            if last_action.get("action") == "done":
                final_result = last_action.get("result", "")

    return {
        "action_history": action_history,
        "actions_taken": len(action_history),
        "final_result": final_result,
        "completion": final_result,
    }
