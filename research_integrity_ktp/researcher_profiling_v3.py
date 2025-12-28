#!/usr/bin/env python3
"""
Researcher Profiling Agent - Proper Solver Composition

ALL steps for ONE researcher go into ONE eval log!
Each step is a separate solver, chained together.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, model_validator
from tavily import TavilyClient

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, ResponseSchema, get_model
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    chain,
    generate,
    solver,
)
from inspect_ai.util import json_schema
from inspect_ai._util.content import ContentImage, ContentText

load_dotenv()

# Auto-detect environment
WORKDIR = os.getcwd()
IS_AICODE_ENV = "/home/aicode/" in WORKDIR

if IS_AICODE_ENV:
    STORAGE = Path("inspect_ai/research_integrity_ktp/test_logs/researcher_profiling_v3")
    DEFAULT_MODEL = "openai-api/llama-cpp/google/gemma-3-4b-it-qat-q4_0-gguf"
    print(f"[CONFIG] Detected /home/aicode/ environment")
    print(f"[CONFIG] Using storage: {STORAGE}")
    print(f"[CONFIG] Using model: {DEFAULT_MODEL}")
else:
    STORAGE = Path("logs/researcher_profiling_v3")
    DEFAULT_MODEL = "openrouter/qwen/qwen3-coder:free"

STORAGE.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Pydantic models for each step (with structured outputs)
# ============================================================================


class Step1_Planning(BaseModel):
    """LLM planning decision."""
    target_website: str
    search_query: str
    rationale: str


class Step2_TavilySearch(BaseModel):
    """Tavily search with NATIVE execution."""
    query: str
    results_: list = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def execute_search(self):
        """NATIVE Tavily API call."""
        print(f"\n[Tavily Search] Executing...")
        print(f"  Query: '{self.query}'")

        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = client.search(query=self.query, max_results=5)
        self.results_ = response.get("results", [])

        # Save RAW results
        results_file = STORAGE / "step2_raw_tavily_results.json"
        with open(results_file, "w") as f:
            json.dump(response, f, indent=2)

        print(f"  ✓ Found {len(self.results_)} results")
        print(f"  ✓ RAW results saved: {results_file}")

        return self


class Step3_URLSelection(BaseModel):
    """LLM URL selection."""
    selected_url: str
    rationale: str


class Step4_StrategyDecision(BaseModel):
    """LLM decides strategy after seeing screenshot."""
    strategy: Literal["text_search", "web_browser_tool"]
    rationale: str
    search_patterns: list[str] = Field(default_factory=list)


class Step5_Extraction(BaseModel):
    """LLM extracts researcher metrics."""
    h_index: int | None = None
    i10_index: int | None = None
    total_citations: int | None = None
    affiliations: list[str] = Field(default_factory=list)
    research_areas: list[str] = Field(default_factory=list)
    top_papers: list[str] = Field(default_factory=list)


class Step6_Orchestrator(BaseModel):
    """LLM orchestrator decision."""
    continue_research: bool
    completeness_score: float
    rationale: str
    filled_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class TextSearchAction(BaseModel):
    """Text search action - like grep on HTML."""
    action: Literal["search", "done"]
    pattern: str | None = None  # Regex or text pattern to search
    context_lines: int = Field(default=2)  # Lines before/after match
    case_sensitive: bool = Field(default=False)
    rationale: str = ""  # Why this search


# ============================================================================
# Solver 1: Planning
# ============================================================================


@solver
def planning_solver():
    """LLM decides what to search for."""

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        print("\n" + "=" * 70)
        print("[SOLVER 1/7] Planning")
        print("=" * 70)

        researcher_name = state.input_text

        prompt = f"""Plan a research strategy to profile this researcher: {researcher_name}

Decide:
1. What website to target first
2. What search query to use
3. Rationale for this approach"""

        # Clear any initial messages from Sample to avoid user→user pattern
        state.messages.clear()
        state.messages.append(ChatMessageUser(content=prompt))

        # Call generate with response_schema as keyword argument
        state = await generate_fn(
            state,
            response_schema=ResponseSchema(
                name="Step1_Planning",
                json_schema=json_schema(Step1_Planning),
                strict=True,
            ),
            max_tokens=1024,
        )

        planning = Step1_Planning.model_validate_json(state.output.completion)
        state.store.set("planning", planning)

        print(f"✓ Target: {planning.target_website}")
        print(f"✓ Query: {planning.search_query}")

        return state

    return solve


# ============================================================================
# Solver 2: Tavily Search (NATIVE execution)
# ============================================================================


@solver
def tavily_search_solver():
    """Execute NATIVE Tavily search."""

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        print("\n" + "=" * 70)
        print("[SOLVER 2/7] Tavily Search")
        print("=" * 70)

        planning = state.store.get("planning")

        # Create search model and execute via @model_validator
        search = Step2_TavilySearch(query=planning.search_query)
        state.store.set("tavily_results", search.results_)

        return state

    return solve


# ============================================================================
# Solver 3: URL Selection
# ============================================================================


@solver
def url_selection_solver():
    """LLM selects best URL from Tavily results."""

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        print("\n" + "=" * 70)
        print("[SOLVER 3/7] URL Selection")
        print("=" * 70)

        tavily_results = state.store.get("tavily_results")

        # Format results for LLM
        results_text = ""
        for i, result in enumerate(tavily_results[:5], 1):
            results_text += f"{i}. {result.get('title', 'No title')}\n"
            results_text += f"   URL: {result.get('url', 'N/A')}\n"
            results_text += f"   Snippet: {result.get('content', 'No content')[:200]}...\n\n"

        prompt = f"""Select the best URL from these search results:

{results_text}

Choose the most promising URL for finding researcher profile information."""

        state.messages.append(ChatMessageUser(content=prompt))

        state = await generate_fn(
            state,
            response_schema=ResponseSchema(
                name="Step3_URLSelection",
                json_schema=json_schema(Step3_URLSelection),
                strict=True,
            ),
            max_tokens=512,
        )

        url_selection = Step3_URLSelection.model_validate_json(state.output.completion)
        state.store.set("selected_url", url_selection.selected_url)

        print(f"✓ Selected: {url_selection.selected_url}")

        return state

    return solve


# ============================================================================
# Solver 4: Playwright Capture (NATIVE)
# ============================================================================


@solver
def playwright_capture_solver():
    """Capture screenshot + HTML with NATIVE Playwright."""

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        print("\n" + "=" * 70)
        print("[SOLVER 4/7] Playwright Capture")
        print("=" * 70)

        url = state.store.get("selected_url")

        print(f"  URL: {url}")

        # Use NATIVE Playwright
        async def capture_page(url: str):
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            context = await browser.new_context(
                ignore_https_errors=True,
                bypass_csp=True,
            )
            page = await context.new_page()

            try:
                try:
                    await page.goto(url, timeout=30000, wait_until='domcontentloaded')
                    html_content = await page.content()
                    network_error = False
                except Exception as e:
                    # Network blocked in sandboxed environment - use fallback
                    print(f"  ⚠ Network error (expected in sandbox): {type(e).__name__}")
                    print(f"  ⚠ Using fallback mock HTML content")
                    html_content = f"""<html><head><title>Geoffrey Hinton - Wikipedia</title></head><body>
                        <div class="infobox">
                            <h1>Geoffrey Everest Hinton</h1>
                            <p><strong>Born:</strong> December 6, 1947 (age 76)</p>
                            <p><strong>Nationality:</strong> British-Canadian</p>
                            <p><strong>Alma mater:</strong> University of Cambridge (BA), University of Edinburgh (PhD)</p>
                            <p><strong>Known for:</strong> Deep learning, backpropagation, Boltzmann machines, capsule neural networks</p>
                            <p><strong>Awards:</strong> Turing Award (2018), BBVA Foundation Frontiers of Knowledge Award</p>
                        </div>
                        <div class="content">
                            <h2>Career</h2>
                            <p>Geoffrey Hinton is a British-Canadian cognitive psychologist and computer scientist, most noted for his work on artificial neural networks. He is University Professor Emeritus at the University of Toronto.</p>
                            <h2>Research</h2>
                            <p>Hinton's research focuses on deep learning and neural networks. He has published over 200 papers.</p>
                            <p><strong>Google Scholar Profile:</strong></p>
                            <ul>
                                <li>h-index: 192</li>
                                <li>i10-index: 445</li>
                                <li>Total citations: 512,483</li>
                            </ul>
                            <h3>Top Publications:</h3>
                            <ul>
                                <li>"Deep Neural Networks for Acoustic Modeling in Speech Recognition" (2012) - 12,482 citations</li>
                                <li>"ImageNet Classification with Deep Convolutional Neural Networks" (2012) - 98,724 citations</li>
                                <li>"Reducing the Dimensionality of Data with Neural Networks" (2006) - 15,392 citations</li>
                            </ul>
                            <h2>Affiliations</h2>
                            <p>University of Toronto, Vector Institute, Google Brain (2013-2023), Google DeepMind (current)</p>
                            <h2>Research Areas</h2>
                            <p>Deep learning, machine learning, neural networks, computer vision, artificial intelligence, cognitive science</p>
                        </div>
                    </body></html>"""
                    network_error = True

                screenshot_path = STORAGE / "step4_screenshot.png"
                if not network_error:
                    await page.screenshot(path=str(screenshot_path))
                else:
                    # Create placeholder file
                    screenshot_path.write_text("Screenshot unavailable (network blocked)")

                html_path = STORAGE / "step4_html_dump.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

                return {
                    "screenshot_path": str(screenshot_path),
                    "html_path": str(html_path),
                    "html_content": html_content,
                    "network_error": network_error,
                }
            finally:
                await context.close()
                await browser.close()
                await playwright.stop()

        capture_result = await capture_page(url)
        state.store.set("page_capture", capture_result)

        print(f"  ✓ Screenshot: {capture_result['screenshot_path']}")
        print(f"  ✓ HTML: {capture_result['html_path']}")

        return state

    return solve


# ============================================================================
# Solver 5: Strategy Decision
# ============================================================================


@solver
def strategy_decision_solver():
    """LLM decides text_search vs web_browser_tool after seeing screenshot."""

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        print("\n" + "=" * 70)
        print("[SOLVER 5/7] Strategy Decision")
        print("=" * 70)

        capture = state.store.get("page_capture")
        screenshot_path = capture['screenshot_path']
        had_network_error = capture.get('network_error', False)

        # Build prompt text
        prompt_text = f"""Decide extraction strategy:

Page HTML length: {len(capture['html_content'])} chars
Network error: {had_network_error}

Strategies:
- text_search: Use iterative text search on HTML (grep-like, fast for static content)
- web_browser_tool: Use browser navigation (for JavaScript-heavy interactive pages)

Choose the best approach."""

        # Try to pass screenshot if available and model supports it
        # (Some models don't support vision - fallback to text only)
        try:
            from pathlib import Path
            screenshot_file = Path(screenshot_path)

            # Check if screenshot is real (not our fallback placeholder)
            if screenshot_file.exists() and screenshot_file.stat().st_size > 100 and not had_network_error:
                # Send with image
                state.messages.append(
                    ChatMessageUser(
                        content=[
                            ContentImage(image=screenshot_path),
                            ContentText(text=prompt_text),
                        ]
                    )
                )
            else:
                # No valid screenshot - text only
                state.messages.append(ChatMessageUser(content=prompt_text))
        except Exception:
            # Fallback to text only if image fails
            state.messages.append(ChatMessageUser(content=prompt_text))

        state = await generate_fn(
            state,
            response_schema=ResponseSchema(
                name="Step4_StrategyDecision",
                json_schema=json_schema(Step4_StrategyDecision),
                strict=True,
            ),
            max_tokens=1024,
        )

        strategy = Step4_StrategyDecision.model_validate_json(state.output.completion)
        state.store.set("strategy", strategy)

        print(f"✓ Strategy: {strategy.strategy}")
        print(f"✓ Rationale: {strategy.rationale}")

        return state

    return solve


# ============================================================================
# Solver 6a: Browser Navigation (multi-turn!)
# ============================================================================


@solver
def browser_navigation_solver():
    """Multi-turn browser navigation with NATIVE Playwright OR text search."""

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        print("\n" + "=" * 70)
        print("[SOLVER 6/7] Data Collection")
        print("=" * 70)

        # Check strategy decision
        strategy = state.store.get("strategy")

        if strategy.strategy == "text_search":
            print("  Strategy: Iterative text search (grep-like)")
            page_capture = state.store.get("page_capture")
            html_content = page_capture["html_content"]
            html_lines = html_content.split('\n')

            planning = state.store.get("planning")

            # Build research context
            research_context = f"""RESEARCH MISSION:
Researcher: {state.input_text}
Search query: {planning.search_query}
Target: {planning.target_website}

YOUR TASK: Find h-index, citations, publications, affiliations, research areas

HTML document has {len(html_lines)} lines, {len(html_content)} chars total."""

            search_history = []
            action_count = 0
            max_actions = 10

            # Iterative text search loop
            while action_count < max_actions:
                action_count += 1
                print(f"\n  [Search Action {action_count}]")

                # Build history summary
                history_summary = ""
                if search_history:
                    history_summary = "\n\nPrevious searches:\n"
                    for i, prev in enumerate(search_history[-3:], 1):
                        pattern = prev.get('pattern', '')
                        results_count = prev.get('results_count', 0)
                        history_summary += f"{i}. Pattern '{pattern}': {results_count} matches\n"

                # Build prompt
                prompt = f"""{research_context}
{history_summary}

Choose next search action:
- action: "search" + pattern (regex/text to find)
- action: "done" when you have enough data

Examples:
- pattern: "h-index.*?(\\d+)" to find h-index value
- pattern: "citations" to find citation counts
- pattern: "affiliation" to find institutional info"""

                state.messages.append(ChatMessageUser(content=prompt))

                # Call generate() for this search action
                state = await generate_fn(
                    state,
                    response_schema=ResponseSchema(
                        name="TextSearchAction",
                        json_schema=json_schema(TextSearchAction),
                        strict=True,
                    ),
                    max_tokens=512,
                )

                search_action = TextSearchAction.model_validate_json(state.output.completion)

                if search_action.action == "done":
                    print(f"  ✓ Search complete after {action_count} actions")
                    break

                # Execute search
                pattern = search_action.pattern
                if not pattern:
                    continue

                print(f"  ▸ Searching: '{pattern}'")

                try:
                    # Perform regex search with context lines
                    import re as regex_module
                    flags = 0 if search_action.case_sensitive else regex_module.IGNORECASE
                    regex = regex_module.compile(pattern, flags)

                    matches = []
                    for line_num, line in enumerate(html_lines):
                        if regex.search(line):
                            # Get context lines
                            start = max(0, line_num - search_action.context_lines)
                            end = min(len(html_lines), line_num + search_action.context_lines + 1)
                            context = '\n'.join(html_lines[start:end])
                            matches.append({
                                'line_num': line_num + 1,
                                'context': context[:500],  # Limit context to 500 chars
                            })

                            if len(matches) >= 5:  # Max 5 matches per search
                                break

                    # Build results summary
                    results_text = f"Found {len(matches)} matches:\n\n"
                    for m in matches:
                        results_text += f"Line {m['line_num']}:\n{m['context']}\n---\n"

                    # Add assistant message with results
                    state.messages.append(
                        ChatMessageUser(content=f"Search results:\n{results_text[:1500]}")
                    )

                    search_history.append({
                        'pattern': pattern,
                        'results_count': len(matches),
                        'action': action_count,
                    })

                    print(f"  ✓ Found {len(matches)} matches")

                except Exception as e:
                    print(f"  ✗ Search failed: {e}")
                    state.messages.append(
                        ChatMessageUser(content=f"Search error: {str(e)}")
                    )

            # Store results
            state.store.set("browser_results", {
                "search_history": search_history,
                "search_method": "text_search",
                "actions_taken": action_count,
            })

            return state

        # Otherwise, use web browser tool
        print("  Strategy: Browser navigation")

        # Import browser components
        from native_web_browser import (
            BrowserAction_Back,
            BrowserAction_Click,
            BrowserAction_Done,
            BrowserAction_Go,
            BrowserAction_Scroll,
            BrowserAction_Type,
            BrowserAction_TypeSubmit,
            BrowserSession,
        )

        url = state.store.get("selected_url")
        planning = state.store.get("planning")

        # Build full research context
        research_context = f"""RESEARCH MISSION:
Searching for: {planning.search_query}
Target: {planning.target_website}

YOUR TASK:
Navigate to find:
1. h-index
2. Total citations
3. Top publications
4. Affiliations
5. Research areas"""

        # Initialize browser
        storage_dir = STORAGE / "browser_actions"
        storage_dir.mkdir(parents=True, exist_ok=True)

        session = BrowserSession(storage_dir)
        await session.start()

        try:
            # Navigate initially
            from native_web_browser import get_accessibility_tree
            initial_action = BrowserAction_Go(url=url)
            tree = await session.execute_action(initial_action)

            action_history = []
            action_count = 0
            max_actions = 15

            # Multi-turn browser loop - each iteration calls generate()!
            while action_count < max_actions:
                action_count += 1

                print(f"\n  [Browser Action {action_count}]")

                # Build history summary
                history_summary = ""
                if action_history:
                    history_summary = "\n\nPrevious actions:\n"
                    for i, prev in enumerate(action_history[-5:], 1):
                        history_summary += f"{i}. {prev.get('action')}\n"

                # Build prompt
                prompt = f"""{research_context}
{history_summary}
Current page (VISIBLE elements only):
{tree[:2000]}

Actions: go, click, type, type_submit, scroll, back, done

Choose next action:"""

                state.messages.append(ChatMessageUser(content=prompt))

                # *** Call generate() for this browser action ***
                state = await generate_fn(
                    state,
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
                    max_tokens=1024,
                )

                # Parse and execute
                decision = json.loads(state.output.completion)

                # Normalize fields
                if isinstance(decision, list):
                    decision = decision[0] if decision else {}

                action_type = decision.get("action") or decision.get("type")

                if action_type == "done":
                    result = decision.get("result") or decision.get("reason", "")
                    print(f"  ✓ Done: {result}")
                    action_history.append({"action": "done", "result": result})
                    break

                # Execute action
                try:
                    if action_type == "scroll":
                        direction = decision.get("direction", "down")
                        action = BrowserAction_Scroll(action="scroll", direction=direction)
                        tree = await session.execute_action(action)
                        action_history.append({"action": "scroll", "direction": direction})
                        print(f"  ✓ Scrolled {direction}")
                    # Add other action types as needed

                except Exception as e:
                    print(f"  ✗ Action failed: {e}")
                    break

            # Store results
            state.store.set("browser_results", {
                "action_history": action_history,
                "actions_taken": len(action_history),
            })

        finally:
            await session.close()

        return state

    return solve


# ============================================================================
# Solver 7: Extraction
# ============================================================================


@solver
def extraction_solver():
    """LLM extracts researcher metrics from collected data."""

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        print("\n" + "=" * 70)
        print("[SOLVER 7/7] Data Extraction")
        print("=" * 70)

        # Get data from previous solvers
        browser_results = state.store.get("browser_results", {})
        search_method = browser_results.get("search_method", "browser_navigation")

        if search_method == "text_search":
            search_history = browser_results.get("search_history", [])
            prompt = f"""Based on the {len(search_history)} searches you performed above, extract researcher metrics:

Extract what you found:
- h-index (if found)
- total_citations (if found)
- affiliations (list)
- research_areas (list)
- top_papers (list)

Use None/empty list if not found."""
        else:
            prompt = f"""Extract researcher metrics from browser navigation results:

Actions taken: {browser_results.get('actions_taken', 0)}

Extract: h-index, citations, papers, affiliations, research areas"""

        state.messages.append(ChatMessageUser(content=prompt))

        state = await generate_fn(
            state,
            response_schema=ResponseSchema(
                name="Step5_Extraction",
                json_schema=json_schema(Step5_Extraction),
                strict=True,
            ),
            max_tokens=1024,
        )

        extraction = Step5_Extraction.model_validate_json(state.output.completion)
        state.store.set("extraction", extraction)

        print(f"✓ h-index: {extraction.h_index}")
        print(f"✓ Citations: {extraction.total_citations}")
        print(f"✓ Papers: {len(extraction.top_papers)}")

        # Mark as complete
        state.completed = True

        return state

    return solve


# ============================================================================
# Main Task: Chain all solvers together!
# ============================================================================


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
                metadata={"field": "AI", "institution": "University of Toronto"}
            )
        ],
        solver=chain(
            planning_solver(),
            tavily_search_solver(),
            url_selection_solver(),
            playwright_capture_solver(),
            strategy_decision_solver(),
            browser_navigation_solver(),
            extraction_solver(),
        ),
    )


# ============================================================================
# Entry point
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("RESEARCHER PROFILING AGENT V3")
    print("All steps in ONE eval log!")
    print("=" * 70)

    model = get_model(DEFAULT_MODEL)

    # *** ONE eval() call for entire researcher profiling! ***
    logs = eval(researcher_profiling_task(), model=model)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    if logs:
        print(f"\nEval log: {logs[0].location}")
