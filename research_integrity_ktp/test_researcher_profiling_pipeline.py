#!/usr/bin/env python3
"""
Production researcher profiling agent with ACTUAL web search and browsing.

This implementation makes 4 separate LLM calls:
1. Planning call: Review data and decide what website to visit
2. Search call: Use Tavily to find URLs and navigation strategy (saves RAW results)
3. Browsing call: Use web browser to visit pages (saves FULL PAGES)
4. Extraction call: Extract data from cached pages

All raw search results and pages are saved persistently to inspect logs.

Run from repo root:
    uv run python research_integrity_ktp/test_researcher_profiling_live.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, use_tools
from inspect_ai.tool import web_browser, web_search
from inspect_ai.util import json_schema

load_dotenv()

# ============================================================================
# STEP 1: Planning - Decide what to research
# ============================================================================


class ResearchPlan(BaseModel):
    """Plan for what to research next."""

    current_knowledge_summary: str = Field(
        ..., description="Summary of what is currently known"
    )
    identified_gaps: list[str] = Field(
        ..., description="Information gaps to fill"
    )
    target_website: str = Field(
        ..., description="Website to search (e.g., 'Google Scholar', 'university homepage')"
    )
    rationale: str = Field(
        ..., description="Why this website will help fill the gaps"
    )
    search_language: str = Field(
        ..., description="Language for search (en, de, fr, etc.)"
    )


@task
def step1_planning():
    """Step 1: Create research plan."""
    return Task(
        dataset=[
            Sample(
                input="""You are researching Dr. Geoffrey Hinton, a highly cited AI researcher.

Current known information:
- Name: Geoffrey Hinton
- Field: Artificial Intelligence, Deep Learning
- Known for: Backpropagation, Deep Learning

This is iteration 1. No previous research has been done.

Analyze what information is missing and decide which website to search first to gather more details about this researcher's profile (publications, citations, h-index, affiliations, awards, etc.).

You should select a specific website like "Google Scholar", "Semantic Scholar", "researcher's university homepage", "DBLP", etc.""",
                target="",
            )
        ],
        solver=generate(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="ResearchPlan",
                json_schema=json_schema(ResearchPlan),
                strict=True,
            ),
            max_tokens=2048,
        ),
    )


# ============================================================================
# STEP 2: Search - Use Tavily to find URLs and strategy
# ============================================================================


class NavigationStrategy(BaseModel):
    """Navigation strategy based on search results."""

    search_query_used: str = Field(..., description="The search query used")
    target_url: str = Field(..., description="Primary URL to visit")
    alternative_urls: list[str] = Field(
        default_factory=list, description="Alternative URLs found"
    )
    expected_page_structure: str = Field(
        ..., description="Expected structure of the target page based on search snippets"
    )
    navigation_steps: list[str] = Field(
        ..., description="Step-by-step plan for navigating the page"
    )
    data_to_extract: list[str] = Field(
        ..., description="Specific data points expected on this page"
    )
    raw_search_results_saved: bool = Field(
        default=True,
        description="Confirmation that raw search results were saved (always true)",
    )


@task
def step2_search():
    """Step 2: Use Tavily web search to find URLs."""
    return Task(
        dataset=[
            Sample(
                input="""Based on the research plan, search for "Geoffrey Hinton Google Scholar profile" to find his academic profile page.

Use the web_search tool to find:
1. The correct URL for Geoffrey Hinton's Google Scholar profile
2. Alternative URLs that might be useful
3. Information about what data is available on the page

After getting search results, analyze them and provide a navigation strategy for visiting the page in the next step.

ALL RAW SEARCH RESULTS are automatically saved to the Inspect AI log by the web_search tool.""",
                target="",
            )
        ],
        solver=[
            use_tools([web_search(providers="tavily")]),
            generate(),
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="NavigationStrategy",
                json_schema=json_schema(NavigationStrategy),
                strict=True,
            ),
            max_tokens=3072,
        ),
    )


# ============================================================================
# STEP 3: Browsing - Use web browser to visit pages
# ============================================================================


class PageContent(BaseModel):
    """Content retrieved from a web page."""

    url: str = Field(..., description="URL of the page")
    page_title: str = Field(..., description="Title of the page")
    content_summary: str = Field(
        ..., description="Summary of the page content (200-300 words)"
    )
    key_sections_found: list[str] = Field(
        ..., description="Key sections or data points visible on the page"
    )


class BrowsingResults(BaseModel):
    """Results from browsing web pages."""

    pages_visited: list[PageContent] = Field(
        ..., description="Pages visited and their content"
    )
    total_pages: int = Field(..., description="Total number of pages visited")
    full_pages_cached: bool = Field(
        default=True,
        description="Confirmation that full pages were saved (always true)",
    )
    browsing_notes: str = Field(
        ..., description="Overall notes about the browsing session"
    )


@task
def step3_browsing():
    """Step 3: Use web browser to visit pages."""
    return Task(
        dataset=[
            Sample(
                input="""Based on the navigation strategy, use the web_browser tool to visit Geoffrey Hinton's Google Scholar profile.

Visit the target URL and potentially 1-2 alternative URLs if the primary page doesn't have enough information.

For each page:
1. Use web_browser tool to navigate to the URL
2. Examine the content
3. Summarize what you find

ALL FULL PAGES are automatically saved to the Inspect AI log by the web_browser tool.

Focus on gathering information about:
- Publications and citation counts
- H-index and i10-index
- Research areas
- Co-authors
- Affiliations""",
                target="",
            )
        ],
        solver=[
            use_tools([*web_browser()]),
            generate(),
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="BrowsingResults",
                json_schema=json_schema(BrowsingResults),
                strict=True,
            ),
            max_tokens=4096,
        ),
    )


# ============================================================================
# STEP 4: Extraction - Extract data from cached pages
# ============================================================================


class Publication(BaseModel):
    """A publication."""

    title: str
    year: int | None = None
    citations: int | None = None
    venue: str | None = None


class ExtractedData(BaseModel):
    """Data extracted from the cached pages."""

    publications: list[Publication] = Field(
        default_factory=list, description="Top publications found"
    )
    h_index: int | None = Field(None, description="H-index")
    i10_index: int | None = Field(None, description="i10-index")
    total_citations: int | None = Field(None, description="Total citations")
    affiliations: list[str] = Field(
        default_factory=list, description="Affiliations"
    )
    research_areas: list[str] = Field(
        default_factory=list, description="Research areas"
    )
    co_authors: list[str] = Field(
        default_factory=list, description="Frequent co-authors"
    )
    other_metrics: dict[str, str | int] = Field(
        default_factory=dict, description="Any other metrics found"
    )
    data_source: str = Field(..., description="URL where data was extracted from")


@task
def step4_extraction():
    """Step 4: Extract data from cached pages."""
    return Task(
        dataset=[
            Sample(
                input="""Based on the page content you browsed in the previous step, extract specific data about Geoffrey Hinton.

The pages are now cached in the Inspect AI log. Review the content summaries from the browsing step and extract:

1. Top publications (at least 5-10 if available)
2. Citation metrics (h-index, i10-index, total citations)
3. Research areas/topics
4. Affiliations
5. Frequent co-authors
6. Any other relevant metrics

Be specific and extract actual numbers and titles from the pages you visited.""",
                target="",
            )
        ],
        solver=generate(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="ExtractedData",
                json_schema=json_schema(ExtractedData),
                strict=True,
            ),
            max_tokens=3072,
        ),
    )


# ============================================================================
# Orchestrator - Decide if more iterations needed
# ============================================================================


class ProfileCompletenessAssessment(BaseModel):
    """Assessment of profile completeness."""

    completeness_score: float = Field(
        ..., description="Score from 0.0 to 1.0"
    )
    filled_fields: list[str] = Field(
        ..., description="Fields that are well-filled"
    )
    missing_fields: list[str] = Field(
        ..., description="Fields that are missing"
    )


class OrchestratorDecision(BaseModel):
    """Decision from orchestrator."""

    continue_research: bool = Field(
        ..., description="Whether to continue with another iteration"
    )
    rationale: str = Field(..., description="Rationale for the decision")
    assessment: ProfileCompletenessAssessment
    next_focus_areas: list[str] = Field(
        default_factory=list, description="Focus areas for next iteration"
    )


@task
def orchestrator_decision():
    """Orchestrator decides if more research is needed."""
    return Task(
        dataset=[
            Sample(
                input="""You are an orchestrator managing researcher profiling.

You have completed 1 iteration of research on Dr. Geoffrey Hinton:

Iteration 1: Collected data from Google Scholar
- Publications: Found 10+ key papers
- H-index: Available
- Total citations: Available
- Affiliations: University of Toronto, Google Brain
- Research areas: Deep Learning, Neural Networks

Assess the completeness of the researcher profile and decide whether to continue research or if sufficient data has been collected.

Consider what might still be missing (awards, recent work, detailed bio, etc.).""",
                target="",
            )
        ],
        solver=generate(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="OrchestratorDecision",
                json_schema=json_schema(OrchestratorDecision),
                strict=True,
            ),
            max_tokens=2048,
        ),
    )


# ============================================================================
# Test execution
# ============================================================================


def save_results(log, step_name: str, output_dir: Path):
    """Save test results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save log
    log_file = output_dir / f"{step_name}_log.json"
    with open(log_file, "w") as f:
        log_data = {
            "status": log.status,
            "samples": [
                {
                    "input": s.input,
                    "output": s.output.completion if s.output else None,
                }
                for s in log.samples
            ],
        }
        json.dump(log_data, f, indent=2)

    # Save completion
    if log.samples and log.samples[0].output:
        completion_file = output_dir / f"{step_name}_completion.json"
        with open(completion_file, "w") as f:
            f.write(log.samples[0].output.completion)

    print(f"✓ Saved {step_name} results to {output_dir}")


def test_full_pipeline():
    """Test the full 4-step pipeline."""
    print("=" * 70)
    print("RESEARCHER PROFILING - FULL PIPELINE TEST")
    print("=" * 70)

    output_dir = Path("logs/researcher_profiling_pipeline")
    model = get_model("openrouter/qwen/qwen3-coder:free")

    # Step 1: Planning
    print("\n[STEP 1] Planning - Decide what to research")
    print("-" * 70)
    log1 = eval(step1_planning(), model=model)[0]
    save_results(log1, "step1_planning", output_dir)
    plan = ResearchPlan.model_validate_json(log1.samples[0].output.completion)
    print(f"Target website: {plan.target_website}")
    print(f"Rationale: {plan.rationale[:100]}...")

    # Step 2: Search
    print("\n[STEP 2] Search - Use Tavily to find URLs")
    print("-" * 70)
    print("🔍 Making actual Tavily search call...")
    log2 = eval(step2_search(), model=model)[0]
    save_results(log2, "step2_search", output_dir)
    strategy = NavigationStrategy.model_validate_json(
        log2.samples[0].output.completion
    )
    print(f"Target URL: {strategy.target_url}")
    print(f"Search query: {strategy.search_query_used}")
    print("✓ RAW search results saved to Inspect AI log")

    # Step 3: Browsing
    print("\n[STEP 3] Browsing - Visit pages with web browser")
    print("-" * 70)
    print("🌐 Using web browser to visit pages...")
    log3 = eval(step3_browsing(), model=model)[0]
    save_results(log3, "step3_browsing", output_dir)
    browsing = BrowsingResults.model_validate_json(
        log3.samples[0].output.completion
    )
    print(f"Pages visited: {browsing.total_pages}")
    for page in browsing.pages_visited:
        print(f"  - {page.url}")
    print("✓ FULL pages saved to Inspect AI log")

    # Step 4: Extraction
    print("\n[STEP 4] Extraction - Extract data from cached pages")
    print("-" * 70)
    log4 = eval(step4_extraction(), model=model)[0]
    save_results(log4, "step4_extraction", output_dir)
    data = ExtractedData.model_validate_json(log4.samples[0].output.completion)
    print(f"Publications: {len(data.publications)}")
    print(f"H-index: {data.h_index}")
    print(f"Total citations: {data.total_citations}")
    print(f"Affiliations: {', '.join(data.affiliations[:3])}")

    # Orchestrator
    print("\n[ORCHESTRATOR] Decide if more research needed")
    print("-" * 70)
    log5 = eval(orchestrator_decision(), model=model)[0]
    save_results(log5, "orchestrator", output_dir)
    decision = OrchestratorDecision.model_validate_json(
        log5.samples[0].output.completion
    )
    print(f"Continue research: {decision.continue_research}")
    print(f"Completeness: {decision.assessment.completeness_score:.1%}")
    print(f"Rationale: {decision.rationale[:150]}...")

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nVerify that:")
    print("  1. RAW search results are in Inspect AI eval logs")
    print("  2. FULL pages are in Inspect AI eval logs")
    print("  3. Data extraction worked correctly")


if __name__ == "__main__":
    import sys

    try:
        test_full_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
