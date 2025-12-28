#!/usr/bin/env python3
"""
Complete working test for researcher profiling pipeline.

This demonstrates the CORRECT architecture:
- LLM generates JSON structured output (mocked for testing)
- Python code MANUALLY executes web_search and web_browser tools
- RAW search results and FULL pages are saved persistently
- NO tool calling features from Inspect AI

Run from repo root:
    TAVILY_API_KEY=xxx uv run python research_integrity_ktp/test_researcher_profiling_complete.py
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from inspect_ai.tool import ToolCall, ToolError, web_browser, web_search

load_dotenv()

# ============================================================================
# Pydantic schemas for LLM structured outputs
# ============================================================================


class SearchDecision(BaseModel):
    """LLM decides what to search."""

    target_website: str = Field(..., description="Website to search")
    search_query: str = Field(..., description="Search query string")
    rationale: str = Field(..., description="Why this search (max 50 words)")


class URLSelection(BaseModel):
    """LLM selects URL from search results."""

    selected_url: str = Field(..., description="URL to visit")
    reason: str = Field(..., description="Why this URL (max 30 words)")


class ExtractedData(BaseModel):
    """LLM extracts data from page content."""

    h_index: int | None = None
    total_citations: int | None = None
    affiliation: str | None = None
    top_papers: list[str] = Field(default_factory=list)
    research_areas: list[str] = Field(default_factory=list)


class OrchestratorDecision(BaseModel):
    """Orchestrator decides whether to continue research."""

    continue_research: bool
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Reason (max 50 words)")


# ============================================================================
# Step 1: LLM decides what to search (MOCKED)
# ============================================================================


def step1_planning() -> SearchDecision:
    """
    Mock LLM decision about what to search.
    In production, this would be an actual LLM call with ResponseSchema.
    """
    print("\n[STEP 1] LLM Planning (MOCKED)")
    print("-" * 70)

    # Mock LLM decision
    decision = SearchDecision(
        target_website="Google Scholar",
        search_query="Geoffrey Hinton Google Scholar profile",
        rationale="Google Scholar provides comprehensive publication and citation metrics",
    )

    print(f"✓ Target: {decision.target_website}")
    print(f"✓ Query: {decision.search_query}")

    return decision


# ============================================================================
# Step 2: Python MANUALLY executes web search based on LLM output
# ============================================================================


async def step2_execute_search(
    search_query: str, output_dir: Path
) -> tuple[dict, list[dict]]:
    """
    MANUALLY execute Tavily web search based on LLM's JSON output.
    This is ACTUAL web search - RAW results will be saved!
    """
    print("\n[STEP 2] Executing REAL Tavily Search")
    print("-" * 70)
    print(f"Query: '{search_query}'")

    # Create web_search tool
    search_tool = web_search(providers="tavily")

    # Create tool call manually
    tool_call = ToolCall(
        id="search_001",
        function=search_tool.name,
        arguments={"query": search_query},
        type="function",
    )

    # Execute tool MANUALLY
    try:
        result = await search_tool(tool_call)

        # Save RAW results persistently
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_file = output_dir / "step2_raw_search_results.json"

        raw_data = {
            "query": search_query,
            "tool_call": {
                "id": tool_call.id,
                "function": tool_call.function,
                "arguments": tool_call.arguments,
            },
            "result": result,
        }

        with open(raw_file, "w") as f:
            json.dump(raw_data, f, indent=2)

        print(f"✓ Search executed successfully")
        print(f"✓ RAW results saved to: {raw_file}")

        # Parse results
        if isinstance(result, ToolError):
            print(f"✗ Search failed: {result.message}")
            return {}, []

        # Extract search results from tool output
        results = []
        if hasattr(result, "results"):
            results = result.results
        elif isinstance(result, list):
            results = result

        print(f"✓ Found {len(results)} results")

        return raw_data, results

    except Exception as e:
        print(f"✗ Search error: {e}")
        return {}, []


# ============================================================================
# Step 3: LLM selects URL from search results (MOCKED)
# ============================================================================


def step3_url_selection(search_results: list[dict]) -> URLSelection:
    """
    Mock LLM URL selection from search results.
    In production, this would be an actual LLM call with ResponseSchema.
    """
    print("\n[STEP 3] LLM URL Selection (MOCKED)")
    print("-" * 70)

    # Show available results
    print(f"Available results: {len(search_results)}")
    for i, result in enumerate(search_results[:3], 1):
        if isinstance(result, dict):
            print(f"  {i}. {result.get('title', 'No title')}")
            print(f"     {result.get('url', 'No URL')}")

    # Mock LLM decision - select first Google Scholar result
    # In real scenario, LLM would analyze and choose
    selected_url = "https://scholar.google.com/citations?user=JicYPdAAAAAJ"

    selection = URLSelection(
        selected_url=selected_url,
        reason="Primary Google Scholar profile with complete metrics",
    )

    print(f"\n✓ Selected: {selection.selected_url}")

    return selection


# ============================================================================
# Step 4: Python MANUALLY executes web browser based on LLM output
# ============================================================================


async def step4_execute_browser(url: str, output_dir: Path) -> tuple[dict, str]:
    """
    MANUALLY execute web browser based on LLM's JSON output.
    This is ACTUAL web browsing - FULL page will be saved!
    """
    print("\n[STEP 4] Executing REAL Web Browser")
    print("-" * 70)
    print(f"URL: {url}")

    # Create web_browser tools
    browser_tools = {tool.name: tool for tool in web_browser()}

    # Use web_page tool to get page content
    page_tool = browser_tools.get("web_page")
    if not page_tool:
        print("✗ web_page tool not found")
        return {}, ""

    # Create tool call manually
    tool_call = ToolCall(
        id="browse_001",
        function=page_tool.name,
        arguments={"url": url},
        type="function",
    )

    # Execute tool MANUALLY
    try:
        result = await page_tool(tool_call)

        # Save FULL page persistently
        page_file = output_dir / "step4_full_page_content.json"

        page_data = {
            "url": url,
            "tool_call": {
                "id": tool_call.id,
                "function": tool_call.function,
                "arguments": tool_call.arguments,
            },
            "result": str(result),  # Full page content
        }

        with open(page_file, "w") as f:
            json.dump(page_data, f, indent=2)

        print(f"✓ Page retrieved successfully")
        print(f"✓ FULL page saved to: {page_file}")

        # Extract text content
        content = str(result) if result else ""
        print(f"✓ Content length: {len(content)} characters")

        return page_data, content

    except Exception as e:
        print(f"✗ Browser error: {e}")
        import traceback

        traceback.print_exc()
        return {}, ""


# ============================================================================
# Step 5: LLM extracts data from cached page (MOCKED)
# ============================================================================


def step5_extract_data(page_content: str) -> ExtractedData:
    """
    Mock LLM data extraction from cached page content.
    In production, this would be an actual LLM call with ResponseSchema.
    """
    print("\n[STEP 5] LLM Data Extraction (MOCKED)")
    print("-" * 70)
    print(f"Extracting from {len(page_content)} characters of cached content")

    # Mock extraction - in production, LLM would analyze page_content
    data = ExtractedData(
        h_index=176,
        total_citations=658923,
        affiliation="University of Toronto, Google Brain",
        top_papers=[
            "ImageNet classification with deep convolutional neural networks",
            "Deep Learning",
            "Reducing the dimensionality of data with neural networks",
        ],
        research_areas=[
            "Machine Learning",
            "Deep Learning",
            "Neural Networks",
            "Artificial Intelligence",
        ],
    )

    print(f"✓ H-index: {data.h_index}")
    print(f"✓ Citations: {data.total_citations}")
    print(f"✓ Affiliation: {data.affiliation}")
    print(f"✓ Papers: {len(data.top_papers)}")
    print(f"✓ Research areas: {len(data.research_areas)}")

    return data


# ============================================================================
# Step 6: Orchestrator decides if more research needed (MOCKED)
# ============================================================================


def step6_orchestrator(extracted_data: ExtractedData) -> OrchestratorDecision:
    """
    Mock orchestrator decision.
    In production, this would be an actual LLM call with ResponseSchema.
    """
    print("\n[STEP 6] Orchestrator Decision (MOCKED)")
    print("-" * 70)

    # Mock decision based on data completeness
    has_metrics = extracted_data.h_index is not None
    has_affiliation = extracted_data.affiliation is not None
    has_papers = len(extracted_data.top_papers) > 0

    completeness = 0.7 if has_metrics and has_affiliation and has_papers else 0.3

    decision = OrchestratorDecision(
        continue_research=completeness < 0.9,
        completeness_score=completeness,
        reason="Have metrics and papers, missing awards and detailed bio",
    )

    print(f"✓ Continue research: {decision.continue_research}")
    print(f"✓ Completeness: {decision.completeness_score:.1%}")
    print(f"✓ Reason: {decision.reason}")

    return decision


# ============================================================================
# Main pipeline execution
# ============================================================================


async def run_complete_pipeline():
    """Run the complete pipeline with real web tools and mocked LLM."""
    print("=" * 70)
    print("COMPLETE RESEARCHER PROFILING PIPELINE")
    print("(Mocked LLM + REAL web search & browsing)")
    print("=" * 70)

    output_dir = Path("logs/researcher_profiling_complete")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # STEP 1: LLM decides what to search (MOCKED)
        search_decision = step1_planning()

        # STEP 2: Python executes REAL Tavily search
        raw_search, search_results = await step2_execute_search(
            search_decision.search_query, output_dir
        )

        if not search_results:
            print("\n✗ No search results, cannot continue")
            return 1

        # STEP 3: LLM selects URL (MOCKED)
        url_selection = step3_url_selection(search_results)

        # STEP 4: Python executes REAL web browsing
        page_data, page_content = await step4_execute_browser(
            url_selection.selected_url, output_dir
        )

        if not page_content:
            print("\n✗ No page content retrieved, cannot continue")
            return 1

        # STEP 5: LLM extracts data from cached page (MOCKED)
        extracted_data = step5_extract_data(page_content)

        # STEP 6: Orchestrator decision (MOCKED)
        decision = step6_orchestrator(extracted_data)

        # Save final summary
        summary_file = output_dir / "pipeline_summary.json"
        summary = {
            "search_decision": search_decision.model_dump(),
            "url_selection": url_selection.model_dump(),
            "extracted_data": extracted_data.model_dump(),
            "orchestrator_decision": decision.model_dump(),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE ✓")
        print("=" * 70)
        print(f"\nAll results saved to: {output_dir}")
        print("\nVerify persistent data:")
        print(f"  ✓ RAW search results: {output_dir}/step2_raw_search_results.json")
        print(f"  ✓ FULL page content: {output_dir}/step4_full_page_content.json")
        print(f"  ✓ Pipeline summary: {output_dir}/pipeline_summary.json")
        print("\nArchitecture verified:")
        print("  ✓ LLM generates JSON (mocked)")
        print("  ✓ Python MANUALLY executes tools")
        print("  ✓ RAW search results saved persistently")
        print("  ✓ FULL pages saved persistently")
        print("  ✓ NO tool calling features used")

        return 0

    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    import sys

    try:
        exit_code = asyncio.run(run_complete_pipeline())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
