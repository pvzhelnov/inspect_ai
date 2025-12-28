#!/usr/bin/env python3
"""
CORRECT researcher profiling implementation:
- LLM generates structured JSON output
- Pydantic validates and AUTOMATICALLY executes native Python functions
- NO tool calling - results flow to next LLM call
- RAW search results and FULL pages saved persistently

Architecture:
1. LLM generates {search_query: "..."} as structured output
2. Pydantic validates JSON and triggers native search function
3. Results saved and passed to next LLM call
4. Repeat for next step

Run from repo root:
    TAVILY_API_KEY=xxx uv run python research_integrity_ktp/test_researcher_profiling_correct_v2.py
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.solver import generate
from inspect_ai.tool import ToolCall, web_browser, web_search
from inspect_ai.util import json_schema

load_dotenv()

# Global storage for results between steps
STORAGE = Path("logs/researcher_profiling_correct_v2")
STORAGE.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helper functions for web operations
# ============================================================================


async def execute_web_search(query: str) -> dict:
    """Native Python function to execute web search."""
    print(f"  🔍 Executing Tavily search: '{query}'")

    search_tool = web_search(providers="tavily")
    tool_call = ToolCall(
        id="search",
        function=search_tool.name,
        arguments={"query": query},
        type="function",
    )

    result = await search_tool(tool_call)

    # Save RAW results
    raw_file = STORAGE / "raw_search_results.json"
    with open(raw_file, "w") as f:
        json.dump(
            {"query": query, "result": str(result)},
            f,
            indent=2,
        )

    print(f"  ✓ RAW results saved to: {raw_file}")

    # Parse results
    results = []
    if hasattr(result, "results"):
        results = result.results
    elif isinstance(result, list):
        results = result

    return {"query": query, "results": results, "count": len(results)}


async def execute_web_browse(url: str) -> dict:
    """Native Python function to execute web browsing."""
    print(f"  🌐 Browsing URL: {url}")

    browser_tools = {tool.name: tool for tool in web_browser()}
    page_tool = browser_tools.get("web_page")

    if not page_tool:
        return {"url": url, "content": "", "error": "web_page tool not found"}

    tool_call = ToolCall(
        id="browse",
        function=page_tool.name,
        arguments={"url": url},
        type="function",
    )

    result = await page_tool(tool_call)
    content = str(result) if result else ""

    # Save FULL page
    page_file = STORAGE / "full_page_content.json"
    with open(page_file, "w") as f:
        json.dump({"url": url, "content": content}, f, indent=2)

    print(f"  ✓ FULL page saved to: {page_file}")
    print(f"  ✓ Content length: {len(content)} characters")

    return {"url": url, "content": content, "length": len(content)}


# ============================================================================
# STEP 1: Search decision with automatic execution
# ============================================================================


class Step1_SearchDecision(BaseModel):
    """
    LLM generates search query.
    Pydantic automatically executes search via model_validator.
    """

    search_query: str = Field(..., description="Search query for web search")
    rationale: str = Field(..., description="Why this query (max 50 words)")

    # Results populated after execution
    search_results: dict = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def execute_search(self):
        """Automatically execute search when model is validated."""
        # This runs AFTER Pydantic validates the JSON from LLM
        print(f"\n[Pydantic Validator] Executing search...")
        self.search_results = asyncio.run(execute_web_search(self.search_query))
        print(f"  ✓ Found {self.search_results.get('count', 0)} results")
        return self


@task
def step1_search_decision():
    """LLM decides what to search."""
    return Task(
        dataset=[
            Sample(
                input="""Researcher: Geoffrey Hinton
Known: AI expert, Deep Learning pioneer
Need: Publications, citations, h-index

Decide search query to find his Google Scholar profile.""",
                target="",
            )
        ],
        solver=generate(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="Step1_SearchDecision",
                json_schema=json_schema(Step1_SearchDecision),
                strict=True,
            ),
            max_tokens=256,
        ),
    )


# ============================================================================
# STEP 2: URL selection (no execution, just decision)
# ============================================================================


class Step2_URLSelection(BaseModel):
    """LLM selects URL from search results."""

    selected_url: str = Field(..., description="URL to visit")
    reason: str = Field(..., description="Why this URL (max 30 words)")


def create_step2_url_selection(search_results: dict):
    """Create task for URL selection."""
    # Format search results for LLM
    results_text = "\n".join(
        [
            f"{i+1}. {r.get('title', 'No title')}\n   URL: {r.get('url', 'No URL')}"
            for i, r in enumerate(search_results.get("results", [])[:5])
        ]
    )

    @task
    def step2_url_selection():
        return Task(
            dataset=[
                Sample(
                    input=f"""Search results:
{results_text}

Select ONE URL to visit for researcher metrics.""",
                    target="",
                )
            ],
            solver=generate(),
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="Step2_URLSelection",
                    json_schema=json_schema(Step2_URLSelection),
                    strict=True,
                ),
                max_tokens=256,
            ),
        )

    return step2_url_selection


# ============================================================================
# STEP 3: Browsing with automatic execution
# ============================================================================


class Step3_BrowseURL(BaseModel):
    """
    LLM confirms URL to browse.
    Pydantic automatically executes browsing via model_validator.
    """

    url: str = Field(..., description="URL to browse")
    expected_data: str = Field(..., description="What data to look for (max 30 words)")

    # Results populated after execution
    page_content: dict = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def execute_browsing(self):
        """Automatically browse page when model is validated."""
        print(f"\n[Pydantic Validator] Executing browser...")
        self.page_content = asyncio.run(execute_web_browse(self.url))
        return self


def create_step3_browse(url: str):
    """Create task for browsing."""

    @task
    def step3_browse():
        return Task(
            dataset=[
                Sample(
                    input=f"""URL to visit: {url}

Confirm the URL and specify what data you expect to find.""",
                    target="",
                )
            ],
            solver=generate(),
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="Step3_BrowseURL",
                    json_schema=json_schema(Step3_BrowseURL),
                    strict=True,
                ),
                max_tokens=256,
            ),
        )

    return step3_browse


# ============================================================================
# STEP 4: Data extraction from cached page
# ============================================================================


class Step4_ExtractedData(BaseModel):
    """LLM extracts data from cached page content."""

    h_index: int | None = None
    total_citations: int | None = None
    affiliation: str | None = None
    top_papers: list[str] = Field(default_factory=list)


def create_step4_extraction(page_content: dict):
    """Create task for data extraction."""
    content_preview = page_content.get("content", "")[:1000]

    @task
    def step4_extract():
        return Task(
            dataset=[
                Sample(
                    input=f"""Page content from {page_content.get('url')}:

{content_preview}...

Extract researcher metrics from this cached page.""",
                    target="",
                )
            ],
            solver=generate(),
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="Step4_ExtractedData",
                    json_schema=json_schema(Step4_ExtractedData),
                    strict=True,
                ),
                max_tokens=512,
            ),
        )

    return step4_extract


# ============================================================================
# Pipeline execution
# ============================================================================


def test_correct_pipeline():
    """Test pipeline with automatic execution via Pydantic validators."""
    print("=" * 70)
    print("CORRECT RESEARCHER PROFILING PIPELINE")
    print("(LLM JSON → Pydantic validates → Native function executes)")
    print("=" * 70)

    model = get_model("openrouter/qwen/qwen3-coder:free")

    # STEP 1: LLM decides search query → Pydantic executes search
    print("\n[STEP 1] LLM generates search query (JSON)")
    print("-" * 70)
    log1 = eval(step1_search_decision(), model=model)[0]
    step1_output = Step1_SearchDecision.model_validate_json(
        log1.samples[0].output.completion
    )
    # ↑ When Pydantic validates, @model_validator executes search automatically!

    print(f"\n✓ LLM Output (JSON):")
    print(f"  search_query: {step1_output.search_query}")
    print(f"  rationale: {step1_output.rationale}")
    print(f"✓ Pydantic executed search automatically")
    print(f"  Results count: {step1_output.search_results.get('count', 0)}")

    # STEP 2: LLM selects URL from results
    print("\n[STEP 2] LLM selects URL from search results (JSON)")
    print("-" * 70)
    step2_task = create_step2_url_selection(step1_output.search_results)
    log2 = eval(step2_task(), model=model)[0]
    step2_output = Step2_URLSelection.model_validate_json(
        log2.samples[0].output.completion
    )

    print(f"✓ LLM Output (JSON):")
    print(f"  selected_url: {step2_output.selected_url}")
    print(f"  reason: {step2_output.reason}")

    # STEP 3: LLM confirms URL → Pydantic executes browsing
    print("\n[STEP 3] LLM confirms browsing (JSON)")
    print("-" * 70)
    step3_task = create_step3_browse(step2_output.selected_url)
    log3 = eval(step3_task(), model=model)[0]
    step3_output = Step3_BrowseURL.model_validate_json(
        log3.samples[0].output.completion
    )
    # ↑ When Pydantic validates, @model_validator executes browsing automatically!

    print(f"\n✓ LLM Output (JSON):")
    print(f"  url: {step3_output.url}")
    print(f"  expected_data: {step3_output.expected_data}")
    print(f"✓ Pydantic executed browsing automatically")
    print(f"  Content length: {step3_output.page_content.get('length', 0)}")

    # STEP 4: LLM extracts data from cached page
    print("\n[STEP 4] LLM extracts data from cached page (JSON)")
    print("-" * 70)
    step4_task = create_step4_extraction(step3_output.page_content)
    log4 = eval(step4_task(), model=model)[0]
    step4_output = Step4_ExtractedData.model_validate_json(
        log4.samples[0].output.completion
    )

    print(f"✓ LLM Output (JSON):")
    print(f"  h_index: {step4_output.h_index}")
    print(f"  total_citations: {step4_output.total_citations}")
    print(f"  affiliation: {step4_output.affiliation}")
    print(f"  top_papers: {len(step4_output.top_papers)}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE ✓")
    print("=" * 70)
    print("\nHow it works:")
    print("  1. LLM generates JSON structured output")
    print("  2. Pydantic validates JSON")
    print("  3. @model_validator automatically calls native Python function")
    print("  4. Results available to next LLM call")
    print("  5. NO tool calling features used!")
    print("\nWhat gets saved:")
    print(f"  ✓ RAW search results: {STORAGE}/raw_search_results.json")
    print(f"  ✓ FULL page content: {STORAGE}/full_page_content.json")


if __name__ == "__main__":
    import sys

    try:
        test_correct_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
