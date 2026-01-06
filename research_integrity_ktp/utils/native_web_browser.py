#!/usr/bin/env python3
"""
Native web browser implementation using Playwright + structured outputs.

Refactored to use the Docker-based web browser implementation from:
inspect_ai/docker/aisiuk/inspect-web-browser-tool/web_browser

Architecture:
1. Launch Playwright browser (using PlaywrightBrowser from docker tool)
2. Create PlaywrightCrawler
3. Navigate to URL, get accessibility tree (using render_at)
4. LLM sees tree + menu of actions
5. LLM outputs structured decision
6. Execute action using crawler methods
7. Get new tree, back to step 4
8. Repeat until LLM outputs "done"
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal, Union

# Add the web_browser directory to sys.path
# We find the relative path from this file
current_file = Path(__file__).resolve()
# Go up 4 levels to reach 'inspect_ai' root from 'inspect_ai/research_integrity_ktp/utils/native_web_browser.py'
project_root = current_file.parents[3]
web_browser_path = project_root / "inspect_ai/docker/aisiuk/inspect-web-browser-tool/web_browser"

if str(web_browser_path) not in sys.path:
    sys.path.append(str(web_browser_path))

try:
    from playwright_browser import PlaywrightBrowser
    from playwright_crawler import PlaywrightCrawler
    # cdp module is inside the web_browser directory, so it should be importable
except ImportError as e:
    raise ImportError(f"Failed to import web_browser modules from {web_browser_path}. Error: {e}")

from pydantic import BaseModel, Field, RootModel, create_model

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.solver import generate
from inspect_ai.util import json_schema, JSONSchemaDict


# ============================================================================ 
# Browser action models with structured outputs
# ============================================================================ 


class BrowserAction_Go(BaseModel):
    """LLM decides to navigate to a URL."""

    reflection: str = Field(..., description="Reasoning for this action")
    action: Literal["go"] = "go"
    url: str = Field(..., description="URL to navigate to")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Click(BaseModel):
    """LLM decides to click an element."""

    reflection: str = Field(..., description="Reasoning for this action")
    action: Literal["click"] = "click"
    element_id: int = Field(..., description="ID of element to click")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Type(BaseModel):
    """LLM decides to type into an element."""

    reflection: str = Field(..., description="Reasoning for this action")
    action: Literal["type"] = "type"
    element_id: int = Field(..., description="ID of element to type into")
    text: str = Field(..., description="Text to type")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_TypeSubmit(BaseModel):
    """LLM decides to type and press ENTER."""

    reflection: str = Field(..., description="Reasoning for this action")
    action: Literal["type_submit"] = "type_submit"
    element_id: int = Field(..., description="ID of element to type into")
    text: str = Field(..., description="Text to type before ENTER")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Scroll(BaseModel):
    """LLM decides to scroll."""

    reflection: str = Field(..., description="Reasoning for this action")
    action: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"] = Field(..., description="Scroll direction")

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Back(BaseModel):
    """LLM decides to go back."""

    reflection: str = Field(..., description="Reasoning for this action")
    action: Literal["back"] = "back"

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Forward(BaseModel):
    """LLM decides to go forward."""

    reflection: str = Field(..., description="Reasoning for this action")
    action: Literal["forward"] = "forward"

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Refresh(BaseModel):
    """LLM decides to refresh the page."""

    reflection: str = Field(..., description="Reasoning for this action")
    action: Literal["refresh"] = "refresh"

    # Result populated after execution
    accessibility_tree_: str = Field(default="", exclude=True)
    screenshot_path_: Path | None = Field(default=None, exclude=True)


class BrowserAction_Done(BaseModel):
    """LLM decides task is complete."""

    reflection: str = Field(..., description="Reasoning for this action")
    action: Literal["done"] = "done"
    result: str = Field(..., description="Final result/answer")


# Union type for all actions
class BrowserAction(RootModel):
    root: Union[
    BrowserAction_Go,
    BrowserAction_Click,
    BrowserAction_Type,
    BrowserAction_TypeSubmit,
    BrowserAction_Scroll,
    BrowserAction_Back,
    BrowserAction_Forward,
    BrowserAction_Refresh,
    BrowserAction_Done,
] = Field(..., discriminator='action')


# ============================================================================ 
# Browser session manager
# ============================================================================ 


class BrowserSession:
    """
    Manages a Playwright browser session using the Docker-based web browser tool implementation.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.browser: PlaywrightBrowser | None = None
        self.context = None
        self.crawler: PlaywrightCrawler | None = None
        self.action_counter = 0

    async def start(self):
        """Launch browser and initialize crawler."""
        # Initialize PlaywrightBrowser (this starts playwright and launches browser)
        self.browser = await PlaywrightBrowser.create(headless=True)
        
        # Create a new context
        self.context = await self.browser.get_new_context()
        
        # Initialize PlaywrightCrawler
        self.crawler = await PlaywrightCrawler.create(self.context)

    async def execute_action(self, action: BrowserAction) -> str:
        """
        Execute browser action and return new accessibility tree.
        """
        if not self.crawler:
            raise RuntimeError("BrowserSession not started. Call start() first.")

        self.action_counter += 1
        page_crawler = await self.crawler.current_page
        
        # Execute action based on type
        if action.action == "go":
            await page_crawler.go_to_url(action.url)

        elif action.action == "click":
            await page_crawler.click(action.element_id)

        elif action.action == "type":
            await page_crawler.type(action.element_id, action.text)

        elif action.action == "type_submit":
            await page_crawler.type(action.element_id, action.text)
            await page_crawler.page.keyboard.press("Enter")
            # Wait for navigation/load if needed, although page_crawler methods usually handle some waiting.
            # Explicitly waiting for network idle can be safer for submissions
            try:
                await page_crawler.page.wait_for_load_state('networkidle', timeout=5000)
            except:
                pass

        elif action.action == "scroll":
            await page_crawler.scroll(action.direction)

        elif action.action == "back":
            await page_crawler.back()

        elif action.action == "forward":
            await page_crawler.forward()

        elif action.action == "refresh":
            await page_crawler.refresh()

        # Update the crawler's view of the page (accessibility tree)
        await page_crawler.update()
        
        # Get accessibility tree and main content
        tree = page_crawler.render_at()
        main_content = page_crawler.render_main_content()

        # Take screenshot using the underlying playwright page
        screenshot_path = self.storage_dir / f"action_{self.action_counter}_screenshot.png"
        await page_crawler.page.screenshot(path=str(screenshot_path))

        # Save HTML dump
        html_path = self.storage_dir / f"action_{self.action_counter}_page.html"
        html_content = await page_crawler.page.content()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Store in action object
        if hasattr(action, 'accessibility_tree_'):
            action.accessibility_tree_ = tree
            action.screenshot_path_ = screenshot_path

        return tree, main_content
        
    async def close(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()


# ============================================================================ 
# Main browsing function with LLM loop
# ============================================================================ 


def browse_with_llm(
    url: str,
    goal: str,
    model,
    response_model: Type[BaseModel] | None = None,
    max_actions: int = 20,
    storage_dir: Path = Path("logs/native_browser"),
) -> dict:
    """
    Browse a URL using LLM + Playwright (Docker implementation).

    Args:
        url: Initial URL to visit
        goal: What to find/accomplish
        model: Inspect AI model
        response_model: Optional Pydantic model for the final result
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

    # Define Dynamic Action Model
    if response_model:
        # Create a new model that combines reflection/action with the user's response model
        fields = {
            "reflection": (str, Field(..., description="Reasoning for this action")),
            "action": (Literal["done"], "done"),
        }
        # Add fields from response_model
        for name, field_info in response_model.model_fields.items():
             fields[name] = (field_info.annotation, field_info)
        
        DynamicDone = create_model(
            "BrowserAction_Done",
            **fields,
        )
    else:
        DynamicDone = BrowserAction_Done

    # Define the Union type dynamically
    ActionUnion = Union[
        BrowserAction_Go,
        BrowserAction_Click,
        BrowserAction_Type,
        BrowserAction_TypeSubmit,
        BrowserAction_Scroll,
        BrowserAction_Back,
        BrowserAction_Forward,
        BrowserAction_Refresh,
        DynamicDone,
    ]
    
    # We use a RootModel to handle the discriminated union
    class DynamicBrowserAction(RootModel):
        root: ActionUnion = Field(..., discriminator='action')

    try:
        # Initial navigation
        initial_action = BrowserAction_Go(url=url, reflection="Initial navigation")
        tree, main_content = loop.run_until_complete(session.execute_action(initial_action))

        action_history = []
        action_count = 0

        while action_count < max_actions:
            action_count += 1

            print(f"\n  [Action {action_count}] Asking LLM for next action...")

            # Build action history summary for context
            history_summary = ""
            if action_history:
                history_summary = "\n\nActions taken so far:\n"
                for i, prev_action in enumerate(action_history[-5:], 1):  # Last 5 actions
                    action_type = prev_action.get("action", "unknown")
                    reflection = prev_action.get("reflection", "No reflection")
                    summary_line = f"{i}. [{action_type}] ({reflection}) "
                    
                    if action_type == "go":
                        summary_line += f"Navigated to: {prev_action.get('url', 'N/A')}"
                    elif action_type == "click":
                        summary_line += f"Clicked element #{prev_action.get('element_id', 'N/A')}"
                    elif action_type == "type":
                        summary_line += f"Typed '{prev_action.get('text', 'N/A')}' into element #{prev_action.get('element_id', 'N/A')}"
                    elif action_type == "type_submit":
                        summary_line += f"Typed and submitted '{prev_action.get('text', 'N/A')}' in element #{prev_action.get('element_id', 'N/A')}"
                    elif action_type == "scroll":
                        summary_line += f"Scrolled {prev_action.get('direction', 'N/A')}"
                    elif action_type == "back":
                        summary_line += "Went back"
                    elif action_type == "forward":
                        summary_line += "Went forward"
                    elif action_type == "refresh":
                        summary_line += "Refreshed page"
                    
                    history_summary += summary_line + "\n"

            # Prepare content context
            content_context = ""
            if main_content:
                content_context += f"Main Content Summary:\n{main_content[:2000]}\n\n"
            content_context += f"Current page accessibility tree:\n{tree[:2000]}  # Truncate for context"

            # Build prompt with action history + current tree + available actions
            prompt = f"""You are browsing a web page to: {goal}
{history_summary}
{content_context}

Available actions:
- go: Navigate to a URL
- click: Click an element by ID
- type: Type text into an element
- type_submit: Type and press ENTER
- scroll: Scroll up or down
- back: Go back
- forward: Go forward
- refresh: Refresh page
- done: Task complete, provide result

Based on your previous actions and the current page, choose your next action.
Always provide a 'reflection' first to explain your reasoning."""

# Your response MUST STRICTLY follow this JSON Schema:

# ```json
# {json.dumps(DynamicBrowserAction.model_json_schema())}
# ```"""

            # Get LLM decision
            @task
            def browser_decision_task():
                return Task(
                    dataset=[Sample(input=prompt, target="")],
                    solver=generate(),
                    config=GenerateConfig(
                        response_schema=ResponseSchema(
                            name="BrowserAction",
                            json_schema=JSONSchemaDict(DynamicBrowserAction.model_json_schema()),
                            strict=True,
                        ),
                        max_tokens=1024, # Increased for reflection
                    ),
                )

            log = eval(browser_decision_task(), model=model)[0]
            completion = log.samples[0].output.completion

            print(f"  DEBUG: LLM completion = {completion[:200]}")

            try:
                decision_json = json.loads(completion)
            except json.JSONDecodeError as e:
                print(f"  ✗ JSON decode error: {e}")
                print(f"  ✗ Completion was: {completion}")
                break

            # Handle invalid types (float, int, str, None)
            if not isinstance(decision_json, (dict, list)):
                print(f"  ✗ LLM returned invalid type: {type(decision_json).__name__}")
                print(f"  ✗ Value: {decision_json}")
                print(f"  ✗ Cannot complete - LLM output is not structured correctly")
                action = BrowserAction_Done(
                    reflection="Error",
                    action="done",
                    result=f"Error: LLM returned {type(decision_json).__name__} instead of action object"
                )
                action_history.append(action.model_dump())
                break

            # Handle if LLM returns a list instead of a single object
            if isinstance(decision_json, list):
                if not decision_json:
                    print(f"  ✗ LLM returned empty list")
                    break
                decision_json = decision_json[0]

            # Flatten if LLM nested parameters inside "parameters" field
            if "parameters" in decision_json and isinstance(decision_json["parameters"], dict):
                params = decision_json.pop("parameters")
                decision_json.update(params)

            # Parse action - handle both "action" and "type" field names
            action_type = decision_json.get("action") or decision_json.get("type")

            # Normalize field names if LLM used "type" instead of "action"
            if "type" in decision_json and "action" not in decision_json:
                decision_json["action"] = decision_json.pop("type")

            # Normalize "query" to "text" for type actions
            if action_type == "type" and "query" in decision_json:
                decision_json["text"] = decision_json.pop("query")

            # Normalize "reason" to "result" for done actions (only if not using custom model)
            if action_type == "done" and not response_model and "reason" in decision_json:
                decision_json["result"] = decision_json.pop("reason")

            # Normalize "element" to "element_id" for click/type actions
            if "element" in decision_json and "element_id" not in decision_json:
                element_val = decision_json.pop("element")
                # Try to convert to int if it's a string number
                try:
                    decision_json["element_id"] = int(element_val)
                except (ValueError, TypeError):
                    decision_json["element_id"] = element_val

            if action_type == "done":
                # Validate against DynamicDone
                try:
                    action = DynamicDone(**decision_json)
                    print(f"  ✓ Task complete.")
                    action_history.append(action.model_dump())
                    break
                except Exception as e:
                    print(f"  ✗ Invalid 'done' action structure: {e}")
                    # Try to fallback to generic done if possible, or just fail
                    action_history.append({"action": "done", "error": str(e), "raw": decision_json})
                    break

            # Create appropriate action object
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
                elif action_type == "forward":
                    action = BrowserAction_Forward(**decision_json)
                elif action_type == "refresh":
                    action = BrowserAction_Refresh(**decision_json)
                else:
                    print(f"  ✗ Unknown action: {action_type}")
                    break
            except Exception as e:
                print(f"  ✗ Invalid action parameters: {e}")
                print(f"  ✗ Cannot complete task - no suitable elements found")
                # Treat as "done" with error message
                action = BrowserAction_Done(
                    reflection="Error",
                    action="done",
                    result=f"Cannot complete: {str(e)}"
                )
                action_history.append(action.model_dump())
                break

            print(f"  ✓ Action: {action.action} (Reflection: {action.reflection})")

            try:
                # Execute action
                tree, main_content = loop.run_until_complete(session.execute_action(action))
                action_history.append(action.model_dump())
            except Exception as e:
                print(f"  ✗ Action execution failed: {e}")
                # Don't break immediately, maybe retry or let LLM try something else?
                # For now, let's record error and continue
                error_action = BrowserAction_Done(
                    reflection="Error execution",
                    action="done", 
                    result=f"Error executing action: {e}"
                )
                action_history.append(error_action.model_dump())
                break

        # Save history
        history_file = storage_dir / "action_history.json"
        with open(history_file, "w") as f:
            json.dump(action_history, f, indent=2)

        print(f"\n  ✓ Session complete, {len(action_history)} actions")
        print(f"  ✓ History saved: {history_file}")
        
        # Determine final result
        last_action = action_history[-1] if action_history else {}
        if last_action.get("action") == "done":
            if response_model:
                # Filter out 'action' and 'reflection' to get just the result model fields
                result_data = {k: v for k, v in last_action.items() if k not in ["action", "reflection"]}
                # We can return the dict, or try to reconstruct the model if needed by caller
                final_result = result_data
            else:
                final_result = last_action.get("result", "")
        else:
            final_result = ""

        return {
            "action_history": action_history,
            "actions_taken": len(action_history),
            "final_result": final_result,
        }

    finally:
        loop.run_until_complete(session.close())
        loop.close()


# ============================================================================ 
# Test function
# ============================================================================ 


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()

    # Test with example.com
    # model = get_model("openai-api/llama-cpp/google/gemma-3-4b-it-qat-q4_0-gguf")
    model = get_model("openai-api/llama-cpp/ggml-org/gpt-oss-20b-GGUF")

    result = browse_with_llm(
        url="http://google.com",
        goal="Search for Geoffrey Hinton and return accurately his place of residence (country, city/town).",
        model=model,
        max_actions=10,
    )

    print("\n" + "=" * 70)
    print("BROWSING COMPLETE")
    print("=" * 70)
    print(f"\nActions taken: {result['actions_taken']}")
    print(f"Final result: {result['final_result']}")