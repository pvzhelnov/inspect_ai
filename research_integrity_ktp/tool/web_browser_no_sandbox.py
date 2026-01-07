#!/usr/bin/env python3
"""
Native (No Sandbox) Web Browser Tool for Inspect AI.

Wraps the Docker-based Playwright implementation directly without using
containers or JSON-RPC, running in the local process.
"""

import asyncio
import re
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Union, Literal

from pydantic import BaseModel, Field

# Inspect AI imports
from inspect_ai import Task, eval, task
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model, ChatMessageUser
from inspect_ai.solver import generate, system_message, use_tools
from inspect_ai.tool import Tool, ToolError, ToolResult, tool, web_search
from inspect_ai.tool._tool_call import ToolCall, ToolCallContent, ToolCallView
from inspect_ai.tool._tool_info import parse_tool_info
from inspect_ai.tool._tool_with import tool_with
from inspect_ai.util._store_model import StoreModel, store_as
from inspect_ai._util.content import ContentText
from inspect_ai.util import json_schema, JSONSchemaDict
from inspect_ai.log import read_eval_log

# ============================================================================ 
# Import Docker-based Browser Tool Implementation
# ============================================================================ 

current_file = Path(__file__).resolve()
# Go up 3 levels from 'inspect_ai/research_integrity_ktp/tool/web_browser_no_sandbox.py'
# to 'inspect_ai' root (which contains 'docker' folder)
project_root = current_file.parents[3] 
web_browser_path = project_root / "inspect_ai/docker/aisiuk/inspect-web-browser-tool/web_browser"

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

if str(web_browser_path) not in sys.path:
    sys.path.append(str(web_browser_path))

try:
    from playwright_browser import PlaywrightBrowser
    from playwright_crawler import PlaywrightCrawler
except ImportError as e:
    logging.warning(f"Could not import web_browser modules: {e}")
    # Define dummy classes if import fails to avoid crash before running
    PlaywrightBrowser = None
    PlaywrightCrawler = None


# ============================================================================ 
# Session Management
# ============================================================================ 

class BrowserSession:
    """Manages a Playwright browser session in-memory."""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser: Optional[PlaywrightBrowser] = None
        self.crawler: Optional[PlaywrightCrawler] = None
        self.context = None

    async def start(self):
        if not PlaywrightBrowser:
            raise RuntimeError("PlaywrightBrowser class not available. Check imports.")
            
        self.browser = await PlaywrightBrowser.create(headless=self.headless)
        self.context = await self.browser.get_new_context()
        self.crawler = await PlaywrightCrawler.create(self.context)

    async def ensure_started(self):
        if not self.browser or not self.crawler:
            await self.start()

    async def close(self):
        if self.browser:
            await self.browser.close()
        self.browser = None
        self.crawler = None
        self.context = None


# Global registry of sessions: instance_id -> BrowserSession
SESSIONS: Dict[str, BrowserSession] = {}

def get_session(instance: str) -> BrowserSession:
    if instance not in SESSIONS:
        SESSIONS[instance] = BrowserSession(headless=True)
    return SESSIONS[instance]


# ============================================================================ 
# Tool Definitions (Copied & Adapted from _web_browser.py)
# ============================================================================ 

def web_browser(
    *, 
    interactive: bool = True, 
    instance: str | None = None,
    headless: bool = True # Added option
) -> list[Tool]:
    """Tools used for web browser navigation.

    To create a separate web browser process for each
    call to `web_browser()`, pass a unique value for `instance`.

    Args:
       interactive: Provide interactive tools (enable
          clicking, typing, and submitting forms). Defaults
          to True.
       instance: Instance id (each unique instance id has its own web browser process)
       headless: Run browser in headless mode. Defaults to True.

    Returns:
       List of tools used for web browser navigation.

    """
    # Ensure session exists with correct settings
    instance_key = instance or "default"
    if instance_key not in SESSIONS:
        SESSIONS[instance_key] = BrowserSession(headless=headless)

    # start with go tool (excluding interactive docs if necessary)
    go = web_browser_go(instance)
    if not interactive:
        go = go_without_interactive_docs(go)
    tools = [go]

    # add interactive tools if requested
    if interactive:
        tools = tools + [
            tool_with_web_at_viewer(web_browser_click(instance), instance),
            tool_with_web_at_viewer(web_browser_type_submit(instance), instance),
            tool_with_web_at_viewer(web_browser_type(instance), instance),
        ]

    # add navigational tools
    return tools + [
        web_browser_scroll(instance),
        web_browser_back(instance),
        web_browser_forward(instance),
        web_browser_refresh(instance),
    ]


@tool(parallel=False)
def web_browser_go(instance: str | None = None) -> Tool:
    """Web Browser tool for navigation to a URL."""
    async def execute(url: str) -> ToolResult:
        """Navigate the web browser to a URL.
        
        Args:
          url (str): URL to navigate to.

        Returns:
          Web accessibility tree of the visible elements of the web page.
        """
        return await _web_browser_cmd("web_go", instance, locals())
    return execute


def go_without_interactive_docs(tool: Tool) -> Tool:
    tool_info = parse_tool_info(tool)
    description_lines = tool_info.description.splitlines()
    description_lines = [
        line for line in description_lines if "web_browser_type_submit" not in line
    ]
    return tool_with(tool, description="\n".join(description_lines))


class WebBrowserStore(StoreModel):
    main_content: str = Field(default_factory=str)
    web_at: str = Field(default_factory=str)
    # session_id not strictly needed for local, but keeping for compatibility
    session_id: str = Field(default_factory=str)


def tool_with_web_at_viewer(tool: Tool, instance: str | None = None) -> Tool:
    def web_at_viewer(call: ToolCall) -> ToolCallView:
        web_at = store_as(WebBrowserStore, instance=instance).web_at
        element_id = call.arguments.get("element_id", 0)
        if web_at and element_id:
            lines = web_at.splitlines()
            pattern = re.compile(rf"^\s+\[{element_id}\] .*$")
            for i, line in enumerate(lines):
                if pattern.match(line):
                    snippet = (
                        lines[0:1]
                        + ["  ..."]
                        + lines[max(i - 2, 1) : i]
                        + [line.replace(" ", "*", 1)]
                        + lines[i + 1 : min(i + 3, len(lines))]
                        + ["  ..."]
                    )
                    return ToolCallView(
                        context=ToolCallContent(
                            format="text", content="\n".join(snippet)
                        )
                    )
        return ToolCallView()
    return tool_with(tool, viewer=web_at_viewer)


@tool(parallel=False)
def web_browser_click(instance: str | None = None) -> Tool:
    """Web Browser tool for clicking an element on a web page."""
    async def execute(element_id: int) -> ToolResult:
        """Click an element on the page.
        
        Args:
           element_id (int): ID of the element to click.
        """
        return await _web_browser_cmd("web_click", instance, locals())
    return execute


@tool(parallel=False)
def web_browser_type_submit(instance: str | None = None) -> Tool:
    """Web Browser tool for typing and submitting input."""
    async def execute(element_id: int, text: str) -> ToolResult:
        """Type text into a form input and press ENTER.
        
        Args:
           element_id (int): ID of the element to type text into.
           text (str): Text to type.
        """
        return await _web_browser_cmd("web_type_submit", instance, locals())
    return execute


@tool(parallel=False)
def web_browser_type(instance: str | None = None) -> Tool:
    """Web Browser tool for typing into inputs."""
    async def execute(element_id: int, text: str) -> ToolResult:
        """Type text into an input.
        
        Args:
           element_id (int): ID of the element to type text into.
           text (str): Text to type.
        """
        return await _web_browser_cmd("web_type", instance, locals())
    return execute


@tool(parallel=False)
def web_browser_scroll(instance: str | None = None) -> Tool:
    """Web Browser tool for scrolling up or down one page."""
    async def execute(direction: str) -> ToolResult:
        """Scroll the web browser up or down by one page.
        
        Args:
           direction (str): "up" or "down"
        """
        return await _web_browser_cmd("web_scroll", instance, locals())
    return execute


@tool(parallel=False)
def web_browser_back(instance: str | None = None) -> Tool:
    """Web Browser tool for navigating back in the browser history."""
    async def execute() -> ToolResult:
        """Navigate the web browser back in the browser history."""
        return await _web_browser_cmd("web_back", instance, locals())
    return execute


@tool(parallel=False)
def web_browser_forward(instance: str | None = None) -> Tool:
    """Web Browser tool for navigating forward in the browser history."""
    async def execute() -> ToolResult:
        """Navigate the web browser forward in the browser history."""
        return await _web_browser_cmd("web_forward", instance, locals())
    return execute


@tool(parallel=False)
def web_browser_refresh(instance: str | None = None) -> Tool:
    """Web Browser tool for refreshing the current page."""
    async def execute() -> ToolResult:
        """Refresh the current page of the web browser."""
        return await _web_browser_cmd("web_refresh", instance, locals())
    return execute


async def _web_browser_cmd(
    tool_name: str, instance: str | None, params: dict[str, object]
) -> ToolResult:
    """
    Execute browser command locally using BrowserSession.
    """
    instance_key = instance or "default"
    session = get_session(instance_key)
    await session.ensure_started()
    
    crawler = session.crawler
    page_crawler = await crawler.current_page
    
    # Map tool names to crawler methods
    try:
        if tool_name == "web_go":
            url = params.get("url")
            await page_crawler.go_to_url(url)
            
        elif tool_name == "web_click":
            element_id = params.get("element_id")
            await page_crawler.click(element_id)
            
        elif tool_name == "web_type":
            element_id = params.get("element_id")
            text = params.get("text")
            await page_crawler.type(element_id, text)
            
        elif tool_name == "web_type_submit":
            element_id = params.get("element_id")
            text = params.get("text")
            await page_crawler.type(element_id, text)
            await page_crawler.page.keyboard.press("Enter")
            try:
                await page_crawler.page.wait_for_load_state('networkidle', timeout=5000)
            except:
                pass
                
        elif tool_name == "web_scroll":
            direction = params.get("direction")
            await page_crawler.scroll(direction)
            
        elif tool_name == "web_back":
            await page_crawler.back()
            
        elif tool_name == "web_forward":
            await page_crawler.forward()
            
        elif tool_name == "web_refresh":
            await page_crawler.refresh()
            
        # Common update after action
        await page_crawler.update()
        
        main_content = page_crawler.render_main_content()
        web_at = page_crawler.render_at() or "(no web accessibility tree available)"
        
        # Filter images from web_at
        web_at_lines = web_at.split("\n")
        web_at_lines = [
            line.partition("data:image/png;base64")[0] for line in web_at_lines
        ]
        web_at = "\n".join(web_at_lines)
        
        # Update store
        store = store_as(WebBrowserStore, instance=instance)
        store.main_content = main_content or "(no main text summary)"
        store.web_at = web_at
        
        return (
            [
                ContentText(text=f"main content:\n{main_content}\n\n"),
                ContentText(text=f"accessibility tree:\n{web_at}")
            ]
            if main_content
            else web_at
        )
        
    except Exception as e:
        raise ToolError(str(e))


# ============================================================================ 
# Main Test Block
# ============================================================================ 

if __name__ == "__main__":
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import inspect

    from dotenv import load_dotenv
    load_dotenv()

    STORAGE = "inspect_ai/research_integrity_ktp/test_logs"

    if str(current_file.parents[2]) not in sys.path:
        sys.path.append(str(current_file.parents[2]))

    from research_integrity_ktp.run import Step2_TavilySearch

    MODEL=get_model("openai-api/llama-cpp/ggml-org/gpt-oss-20b-GGUF")
    #MODEL=get_model("openai-api/llama-cpp/google/gemma-3-4b-it-qat-q4_0-gguf")
    #MODEL=get_model("openai-api/llama-cpp/google/gemma-3-12b-it-qat-q4_0-gguf")
    
    # 1. Define Structured Output Model for Final Result
    class PageSummary(BaseModel):
        reasoning_setting: Literal["high"]
        reflection: str = Field(..., description="Reflection as to the task and context available, how to solve it, etc. It must be as long and as thorough as dictated by the `reasoning_setting`.")
        country: str | None = Field(..., description="country 2-letter code in ISO format")
        location: tuple[float, float] | tuple[None, None] = Field(..., description="approximate but accurate gps coordinates of the exact place")

    @tool
    def native_tavily_search():
        async def execute(query: str):
            """
            Search for query in Tavily API.

            Args:
                query: Search query

            Returns:
                List of strings containing results.
            """
            step_2_tavily_search = Step2_TavilySearch(query=query)
            return step_2_tavily_search.results_
        return execute
        
    # 2. Define the Task
    @task
    def browser_test_task():
        return Task(
            dataset=[Sample(
                input=f"Search for Geoffrey Hinton and return accurately his place of residence (country, city/town). Note that we are not interested in private data, rather we are interested where they work, so the location of work. We are interested in CURRENT workplace, as of {datetime.now(ZoneInfo("America/Toronto")).year}.",
                target=""
            )],
            solver=[
                system_message("IF THE ANSWER IS NOT IN THE CONTEXT, NEVER INFER OR ASSUME. ALWAYS ONLY STATE BASED ON THE INFORMATION AVAILABLE IN THE CONTEXT."),
                use_tools(web_browser(headless=True), native_tavily_search()),
                generate()
            ],
            model=MODEL,
            config=GenerateConfig(max_tokens=1024),
        )

    print("\n" + "="*50)
    print("Running Browser Tool Test (No Sandbox)")
    print("="*50)

    # 3. Run Eval
    #logs = eval(browser_test_task(), limit=1)
    logs = [read_eval_log("/Volumes/home/aicode/research_integrity_ktp_agent/logs/2026-01-07T19-59-56+00-00_browser-test-task_EAMZgFP6pmgihDv4fSeEP6.eval")]

    
    # 4. Extract Final Result from Log and Structure it
    if logs and logs[0].samples:
        log = logs[0]
        sample_result = log.samples[0].output.completion
        print(f"\n[Raw Agent Output]:\n{sample_result}\n")

        print("-" * 50)
        print("Structuring Result with LLM...")

        prompt = f"""Extract structured data from the following text snippet:

```text
{sample_result}
```

You may use previous conversation history as context for the snippet.

Your response must be a JSON and nothing else, valid under the following Pydantic model:

```python
{inspect.getsource(PageSummary)}
```
"""

        @agent
        def sgr_agent() -> Agent:
            async def execute(state: AgentState) -> AgentState:
                state.messages.extend([msg for msg in log.samples[0].messages if msg.role in ("user", "assistant")])
                
                state.messages.append(ChatMessageUser(content=prompt))

                # run a tool loop w/ the web_browser then update & return state
                messages, state.output = await get_model().generate_loop(
                    input=state.messages,
                )

                state.messages.extend(messages)
                return state

            return execute
        
        @task
        def researcher_profiling_task():
            """
            Complete researcher profiling in ONE eval log!
            Each step is a separate solver, all chained together.
            """
            return Task(
                dataset=[
                    Sample(
                        input="Geoffrey Hinton",
                    )
                ],
                solver=sgr_agent(),
            )

        struct_logs = eval(researcher_profiling_task(), model=MODEL)
        structured_output = struct_logs[0].samples[0].output.completion
        
        print(f"\n[Structured Result]:\n{structured_output}")
    else:
        print("No results returned.")

    # Cleanup sessions
    async def cleanup():
        for session in SESSIONS.values():
            await session.close()
            
    asyncio.run(cleanup())
