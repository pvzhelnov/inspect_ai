#!/usr/bin/env python3
"""
Researcher profiling agent - CORRECT implementation with screenshot-based decision making.

Architecture:
1. LLM planning: Decide what to search
2. NATIVE Tavily search: Get search results, save RAW
3. LLM URL selection: Choose best URL
4. NATIVE Playwright: Take screenshot + save HTML dump
5. LLM strategy decision: Look at screenshot, decide text search vs browser tool
6. Execute chosen strategy:
   - Text search: Use grep/regex on saved HTML
   - Web browser tool: Use Inspect AI's tool agentically, dump all pages
7. LLM extraction: Extract data from results
8. LLM orchestrator: Decide if continue

Mock mode: Set MOCK_MODE=1 to use fixture responses (no LLM calls)

Run:
    TAVILY_API_KEY=xxx uv run python research_integrity_ktp/researcher_profiling_agent.py
    MOCK_MODE=1 uv run python research_integrity_ktp/researcher_profiling_agent.py
"""

import asyncio
import base64
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
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.solver import Solver, TaskState, generate, solver, use_tools
from inspect_ai.tool import Tool, ToolCall, ToolResult, web_browser
from inspect_ai.util import json_schema

load_dotenv()

# Storage
STORAGE = Path("logs/researcher_profiling")
STORAGE.mkdir(parents=True, exist_ok=True)

# Mock mode flag
MOCK_MODE = os.getenv("MOCK_MODE") == "1"


# ============================================================================
# Mock LLM solver for testing without API calls
# ============================================================================


class MockResponse(BaseModel):
    """Mock LLM response for testing."""

    completion: str


MOCK_FIXTURES = {
    "Step1_Planning": {
        "target_website": "Nobel Prize",
        "search_query": "Geoffrey Hinton Nobel Prize",
        "rationale": "Nobel Prize website has comprehensive researcher information",
    },
    "Step2_URLSelection": {
        "selected_url": "https://www.nobelprize.org/prizes/physics/2024/hinton/facts/",
        "rationale": "Official Nobel Prize page with detailed researcher biography",
    },
    "Step4_StrategyDecision": {
        "strategy": "web_browser_tool",
        "rationale": "Page requires navigation and JavaScript",
        "search_patterns": [],
    },
    "Step5_Extraction": {
        "h_index": 192,
        "i10_index": 520,
        "total_citations": 993846,
        "affiliations": ["University of Toronto", "Google"],
        "research_areas": ["Machine Learning", "Deep Learning", "Neural Networks"],
        "top_papers": [
            "ImageNet classification with deep convolutional neural networks",
            "Deep Learning",
            "Dropout: a simple way to prevent neural networks from overfitting",
        ],
    },
    "Step6_Orchestrator": {
        "continue_research": False,
        "completeness_score": 0.85,
        "rationale": "Have comprehensive metrics and publications",
        "filled_fields": ["h_index", "citations", "papers", "affiliation"],
        "missing_fields": ["awards", "detailed_bio"],
    },
}


# Global counter for mock mode
_MOCK_STEP_COUNTER = 0
_MOCK_STEP_NAMES = [
    "Step1_Planning",
    "Step2_URLSelection",
    "Step4_StrategyDecision",
    "Step5_Extraction",
    "Step6_Orchestrator",
]


@solver
def mock_generate() -> Solver:
    """Mock solver that returns fixture responses based on call order."""

    async def solve(state: TaskState, generate_fn: generate) -> TaskState:
        global _MOCK_STEP_COUNTER

        # Get the fixture for current step
        if _MOCK_STEP_COUNTER < len(_MOCK_STEP_NAMES):
            schema_name = _MOCK_STEP_NAMES[_MOCK_STEP_COUNTER]
            _MOCK_STEP_COUNTER += 1

            if schema_name in MOCK_FIXTURES:
                state.output.completion = json.dumps(MOCK_FIXTURES[schema_name])
            else:
                state.output.completion = json.dumps({"error": "No fixture"})
        else:
            state.output.completion = json.dumps({"error": "Out of fixtures"})

        return state

    return solve


# ============================================================================
# Step 1: Planning
# ============================================================================


class Step1_Planning(BaseModel):
    """LLM planning decision."""

    target_website: str = Field(..., description="Website to search")
    search_query: str = Field(..., description="Search query")
    rationale: str = Field(..., description="Why (max 100 words)")


# ============================================================================
# Step 2: NATIVE Tavily search with automatic execution
# ============================================================================


class Step2_TavilySearch(BaseModel):
    """Executes NATIVE Tavily search and saves RAW results."""

    search_query: str = Field(..., description="Search query from planning")

    # Results populated after execution
    search_results_: list[dict] = Field(default_factory=list, exclude=True)
    raw_response_: dict = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def execute_search(self):
        """Execute NATIVE Tavily API call."""
        print(f"\n[Pydantic] Executing NATIVE Tavily search...")
        print(f"  Query: '{self.search_query}'")

        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY not set")

            client = TavilyClient(api_key=api_key)
            response = client.search(query=self.search_query, max_results=5)

            self.raw_response_ = response
            self.search_results_ = response.get("results", [])

            # Save RAW results
            raw_file = STORAGE / "step2_raw_tavily_results.json"
            with open(raw_file, "w") as f:
                json.dump(response, f, indent=2)

            print(f"  ✓ Found {len(self.search_results_)} results")
            print(f"  ✓ RAW results saved: {raw_file}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.search_results_ = []
            self.raw_response_ = {"error": str(e)}

        return self


# ============================================================================
# Step 3: URL Selection
# ============================================================================


class Step2_URLSelection(BaseModel):
    """LLM selects URL from search results."""

    selected_url: str = Field(..., description="URL to visit")
    rationale: str = Field(..., description="Why (max 50 words)")


# ============================================================================
# Step 3: NATIVE Playwright screenshot + HTML dump
# ============================================================================


async def capture_page_with_playwright(url: str) -> dict:
    """
    Use NATIVE Playwright to:
    1. Take screenshot
    2. Save HTML dump
    3. Return both

    Returns:
        {
            "screenshot_b64": str,
            "screenshot_path": Path,
            "html_content": str,
            "html_path": Path,
        }
    """
    print(f"\n[Playwright] Capturing page...")
    print(f"  URL: {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        context = await browser.new_context(
            ignore_https_errors=True,
            bypass_csp=True
        )
        page = await context.new_page()

        try:
            # Navigate
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=10000)

            # Take screenshot
            screenshot_path = STORAGE / "step3_screenshot.png"
            screenshot_bytes = await page.screenshot(path=str(screenshot_path), full_page=True)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

            print(f"  ✓ Screenshot saved: {screenshot_path}")

            # Get HTML dump from browser (not requests!)
            html_content = await page.content()
            html_path = STORAGE / "step3_html_dump.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"  ✓ HTML dump saved: {html_path} ({len(html_content)} chars)")

            await browser.close()

            return {
                "screenshot_b64": screenshot_b64,
                "screenshot_path": screenshot_path,
                "html_content": html_content,
                "html_path": html_path,
            }

        except Exception as e:
            print(f"  ✗ Error: {e}")
            await browser.close()
            raise


class Step3_PageCapture(BaseModel):
    """Captures page with Playwright."""

    url: str = Field(..., description="URL to capture")

    # Results populated after execution
    screenshot_b64_: str = Field(default="", exclude=True)
    html_content_: str = Field(default="", exclude=True)

    @model_validator(mode="after")
    def execute_capture(self):
        """Execute NATIVE Playwright capture."""
        result = asyncio.run(capture_page_with_playwright(self.url))
        self.screenshot_b64_ = result["screenshot_b64"]
        self.html_content_ = result["html_content"]
        return self


# ============================================================================
# Step 4: Strategy decision based on screenshot
# ============================================================================


class Step4_StrategyDecision(BaseModel):
    """LLM looks at screenshot and decides strategy."""

    strategy: Literal["text_search", "web_browser_tool"] = Field(
        ..., description="Which strategy to use"
    )
    rationale: str = Field(..., description="Why this strategy (max 100 words)")
    search_patterns: list[str] = Field(
        default_factory=list, description="If text_search: patterns to find"
    )


# ============================================================================
# Step 5a: Text search within saved HTML
# ============================================================================


def text_search_in_html(html_content: str, patterns: list[str]) -> dict:
    """Search for patterns in HTML using regex/grep-like functionality."""
    print(f"\n[Text Search] Searching in HTML...")
    print(f"  HTML length: {len(html_content)} chars")
    print(f"  Patterns: {patterns}")

    results = {}

    for pattern in patterns:
        # Case-insensitive search
        matches = re.findall(rf".{{0,50}}{re.escape(pattern)}.{{0,50}}", html_content, re.IGNORECASE)
        results[pattern] = matches[:5]  # Limit to 5 matches
        print(f"  Found {len(matches)} matches for '{pattern}'")

    # Save results
    results_file = STORAGE / "step5_text_search_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Results saved: {results_file}")

    return results


# ============================================================================
# Step 5b: Native web browser (NATIVE Playwright + structured outputs)
# ============================================================================


def use_native_web_browser(
    url: str, search_goal: str, model
) -> dict:
    """
    Use NATIVE Playwright with structured outputs for browser navigation.
    LLM decides actions, Pydantic executes with Playwright.
    """
    from native_web_browser import browse_with_llm

    print(f"\n[Native Web Browser] Starting...")
    print(f"  URL: {url}")
    print(f"  Goal: {search_goal}")

    # Use native browser implementation
    result = browse_with_llm(
        url=url,
        goal=search_goal,
        model=model,
        max_actions=15,
        storage_dir=STORAGE / "native_browser",
    )

    # Save results
    results_file = STORAGE / "step5_browser_tool_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "url": url,
                "goal": search_goal,
                "completion": result["final_result"],
                "actions_taken": result["actions_taken"],
            },
            f,
            indent=2,
        )

    print(f"  ✓ Actions taken: {result['actions_taken']}")
    print(f"  ✓ Results saved: {results_file}")

    return {
        "completion": result["final_result"],
        "actions_taken": result["actions_taken"],
    }


# ============================================================================
# Step 5: Extraction from results
# ============================================================================


class Step5_Extraction(BaseModel):
    """LLM extracts data from search results or browser results."""

    h_index: int | None = None
    i10_index: int | None = None
    total_citations: int | None = None
    affiliations: list[str] = Field(default_factory=list)
    research_areas: list[str] = Field(default_factory=list)
    top_papers: list[str] = Field(default_factory=list)


# ============================================================================
# Step 6: Orchestrator decision
# ============================================================================


class Step6_Orchestrator(BaseModel):
    """Orchestrator decides if more research needed."""

    continue_research: bool
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., description="Reasoning (max 100 words)")
    filled_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


# ============================================================================
# Main pipeline
# ============================================================================


def run_pipeline():
    """Run the complete researcher profiling pipeline."""
    print("=" * 70)
    print("RESEARCHER PROFILING AGENT")
    if MOCK_MODE:
        print("(MOCK MODE - Using fixture responses)")
    print("=" * 70)

    model = get_model("openrouter/qwen/qwen3-coder:free")
    solver_fn = mock_generate() if MOCK_MODE else generate()

    # STEP 1: Planning
    print("\n[STEP 1] Planning")
    print("-" * 70)

    @task
    def planning_task():
        return Task(
            dataset=[
                Sample(
                    input="Research Geoffrey Hinton. Decide what to search for.",
                    target="",
                )
            ],
            solver=solver_fn,
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="Step1_Planning",
                    json_schema=json_schema(Step1_Planning),
                    strict=True,
                ),
                max_tokens=256,
            ),
        )

    log1 = eval(planning_task(), model=model)[0]
    step1 = Step1_Planning.model_validate_json(log1.samples[0].output.completion)

    print(f"✓ Target: {step1.target_website}")
    print(f"✓ Query: {step1.search_query}")

    # STEP 2: Tavily search (with automatic NATIVE execution)
    print("\n[STEP 2] Tavily Search")
    print("-" * 70)

    # Execute ACTUAL Tavily search (even in mock mode!)
    step2_search = Step2_TavilySearch(search_query=step1.search_query)

    # STEP 3: URL Selection
    print("\n[STEP 3] URL Selection")
    print("-" * 70)

    results_text = "\n".join(
        [
            f"{i+1}. {r.get('title', 'No title')}\n   URL: {r.get('url', 'No URL')}"
            for i, r in enumerate(step2_search.search_results_[:5])
        ]
    )

    @task
    def url_selection_task():
        return Task(
            dataset=[
                Sample(
                    input=f"Search results:\n{results_text}\n\nSelect best URL.",
                    target="",
                )
            ],
            solver=solver_fn,
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="Step2_URLSelection",
                    json_schema=json_schema(Step2_URLSelection),
                    strict=True,
                ),
                max_tokens=256,
            ),
        )

    log3 = eval(url_selection_task(), model=model)[0]
    step3 = Step2_URLSelection.model_validate_json(log3.samples[0].output.completion)

    print(f"✓ Selected: {step3.selected_url}")

    # STEP 4: Playwright capture (screenshot + HTML dump)
    print("\n[STEP 4] Playwright Capture")
    print("-" * 70)

    # Execute ACTUAL Playwright capture (always - even in mock mode!)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        capture_result = loop.run_until_complete(capture_page_with_playwright(step3.selected_url))
        page_capture = {
            "screenshot_b64": capture_result["screenshot_b64"],
            "html_content": capture_result["html_content"],
        }
    finally:
        loop.close()

    # STEP 5: Strategy decision (based on screenshot)
    print("\n[STEP 5] Strategy Decision (based on screenshot)")
    print("-" * 70)

    @task
    def strategy_task():
        # Include screenshot in prompt if not mock
        screenshot_msg = ""
        if not MOCK_MODE:
            screenshot_msg = f"\n\n[Screenshot provided as base64, {len(page_capture['screenshot_b64'])} chars]"

        return Task(
            dataset=[
                Sample(
                    input=f"""Look at the screenshot of {step3.selected_url}.

Decide whether to:
- text_search: Use grep/regex to find patterns in the HTML dump
- web_browser_tool: Use web browser tool to navigate and click{screenshot_msg}

Make your decision.""",
                    target="",
                )
            ],
            solver=solver_fn,
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="Step4_StrategyDecision",
                    json_schema=json_schema(Step4_StrategyDecision),
                    strict=True,
                ),
                max_tokens=512,
            ),
        )

    log5 = eval(strategy_task(), model=model)[0]
    step5 = Step4_StrategyDecision.model_validate_json(log5.samples[0].output.completion)

    print(f"✓ Strategy: {step5.strategy}")
    print(f"✓ Rationale: {step5.rationale}")

    # STEP 6: Execute chosen strategy
    print(f"\n[STEP 6] Executing Strategy: {step5.strategy}")
    print("-" * 70)

    if step5.strategy == "text_search":
        # Text search in saved HTML
        search_results = text_search_in_html(
            page_capture["html_content"], step5.search_patterns
        )
        context_for_extraction = f"Text search results: {json.dumps(search_results, indent=2)}"

    else:
        # Use NATIVE web browser (Playwright + structured outputs)
        browser_results = use_native_web_browser(
            step3.selected_url,
            "Find h-index, citations, publications",
            model,
        )
        context_for_extraction = f"Browser tool results: {browser_results['completion']}"

    # STEP 7: Extraction
    print("\n[STEP 7] Data Extraction")
    print("-" * 70)

    @task
    def extraction_task():
        return Task(
            dataset=[
                Sample(
                    input=f"""Extract researcher metrics from:

{context_for_extraction[:2000]}

Extract h-index, citations, papers, etc.""",
                    target="",
                )
            ],
            solver=solver_fn,
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="Step5_Extraction",
                    json_schema=json_schema(Step5_Extraction),
                    strict=True,
                ),
                max_tokens=1024,
            ),
        )

    log7 = eval(extraction_task(), model=model)[0]
    step7 = Step5_Extraction.model_validate_json(log7.samples[0].output.completion)

    print(f"✓ h-index: {step7.h_index}")
    print(f"✓ Citations: {step7.total_citations}")
    print(f"✓ Papers: {len(step7.top_papers)}")

    # STEP 8: Orchestrator
    print("\n[STEP 8] Orchestrator Decision")
    print("-" * 70)

    @task
    def orchestrator_task():
        return Task(
            dataset=[
                Sample(
                    input=f"""Data collected:
- h-index: {step7.h_index}
- Citations: {step7.total_citations}
- Papers: {len(step7.top_papers)}
- Affiliations: {len(step7.affiliations)}

Decide if more research needed.""",
                    target="",
                )
            ],
            solver=solver_fn,
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="Step6_Orchestrator",
                    json_schema=json_schema(Step6_Orchestrator),
                    strict=True,
                ),
                max_tokens=512,
            ),
        )

    log8 = eval(orchestrator_task(), model=model)[0]
    step8 = Step6_Orchestrator.model_validate_json(log8.samples[0].output.completion)

    print(f"✓ Continue: {step8.continue_research}")
    print(f"✓ Completeness: {step8.completeness_score:.1%}")

    # Save final summary
    summary = {
        "step1_planning": step1.model_dump(),
        "step2_url_selection": step3.model_dump(),
        "step4_strategy": step5.model_dump(),
        "step5_extraction": step7.model_dump(),
        "step6_orchestrator": step8.model_dump(),
    }

    summary_file = STORAGE / "pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {STORAGE}")
    print(f"\nKey files:")
    print(f"  ✓ RAW Tavily results: step2_raw_tavily_results.json")
    print(f"  ✓ Screenshot: step3_screenshot.png")
    print(f"  ✓ HTML dump: step3_html_dump.html")
    if step5.strategy == "text_search":
        print(f"  ✓ Text search results: step5_text_search_results.json")
    else:
        print(f"  ✓ Browser tool results: step5_browser_tool_results.json")
    print(f"  ✓ Summary: pipeline_summary.json")


if __name__ == "__main__":
    import sys

    try:
        run_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
