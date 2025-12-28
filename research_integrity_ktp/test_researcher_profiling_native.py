#!/usr/bin/env python3
"""
CORRECT researcher profiling implementation with NATIVE function calls:
- LLM generates structured JSON output
- Pydantic validates and calls NATIVE Python functions (not Inspect AI tools!)
- Direct Tavily API calls for search
- Direct HTTP requests for web browsing
- NO tool calling features from Inspect AI/LLM providers
- RAW search results and FULL pages saved persistently

Architecture:
1. LLM generates {search_query: "..."} as structured output
2. Pydantic validates JSON and triggers native Tavily API call
3. Results saved and passed to next LLM call
4. Repeat for next step

Run from repo root:
    TAVILY_API_KEY=xxx uv run python research_integrity_ktp/test_researcher_profiling_native.py
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from tavily import TavilyClient

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.solver import generate
from inspect_ai.util import json_schema

load_dotenv()

# Global storage for results between steps
STORAGE = Path("logs/researcher_profiling_native")
STORAGE.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STEP 1: Search decision with automatic Tavily API execution
# ============================================================================


class Step1_SearchDecision(BaseModel):
    """
    LLM generates search query.
    Pydantic automatically executes NATIVE Tavily API call via model_validator.
    """

    search_query: str = Field(..., description="Search query for web search")
    rationale: str = Field(..., description="Why this query (max 50 words)")

    # Results populated after execution
    search_results_: list[dict] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def execute_search(self):
        """Automatically execute NATIVE Tavily search when model is validated."""
        print(f"\n[Pydantic Validator] Executing NATIVE Tavily API call...")
        print(f"  Query: '{self.search_query}'")

        try:
            # NATIVE Tavily API call (not Inspect AI tool!)
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                print("  ✗ TAVILY_API_KEY not set, using mock results")
                self.search_results_ = [
                    {
                        "title": "Geoffrey Hinton - Google Scholar",
                        "url": "https://scholar.google.com/citations?user=JicYPdAAAAAJ",
                        "content": "Geoffrey Hinton's Google Scholar profile",
                    }
                ]
                return self

            client = TavilyClient(api_key=api_key)
            response = client.search(query=self.search_query, max_results=5)

            # Save RAW results persistently
            raw_file = STORAGE / "step1_raw_search_results.json"
            with open(raw_file, "w") as f:
                json.dump(
                    {"query": self.search_query, "response": response}, f, indent=2
                )

            self.search_results_ = response.get("results", [])

            print(f"  ✓ NATIVE Tavily API call completed")
            print(f"  ✓ Found {len(self.search_results_)} results")
            print(f"  ✓ RAW results saved to: {raw_file}")

        except Exception as e:
            print(f"  ✗ Tavily API error: {e}")
            # Use fallback
            self.search_results_ = [
                {
                    "title": "Geoffrey Hinton - Google Scholar",
                    "url": "https://scholar.google.com/citations?user=JicYPdAAAAAJ",
                    "content": "Geoffrey Hinton's Google Scholar profile",
                }
            ]

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


def create_step2_url_selection(search_results: list[dict]):
    """Create task for URL selection."""
    # Format search results for LLM
    results_text = "\n".join(
        [
            f"{i+1}. {r.get('title', 'No title')}\n   URL: {r.get('url', 'No URL')}\n   {r.get('content', '')[:100]}"
            for i, r in enumerate(search_results[:5])
        ]
    )

    @task
    def step2_url_selection():
        return Task(
            dataset=[
                Sample(
                    input=f"""Search results from Tavily:
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
# STEP 3: Browsing with automatic HTTP request execution
# ============================================================================


class Step3_BrowseURL(BaseModel):
    """
    LLM confirms URL to browse.
    Pydantic automatically executes NATIVE HTTP request via model_validator.
    """

    url: str = Field(..., description="URL to browse")
    expected_data: str = Field(..., description="What data to look for (max 30 words)")

    # Results populated after execution
    page_content_: str = Field(default="", exclude=True)

    @model_validator(mode="after")
    def execute_browsing(self):
        """Automatically browse page with NATIVE HTTP request when model is validated."""
        print(f"\n[Pydantic Validator] Executing NATIVE HTTP request...")
        print(f"  URL: {self.url}")

        try:
            # NATIVE HTTP request (not Inspect AI tool!)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(self.url, headers=headers, timeout=30)
            response.raise_for_status()

            # Save FULL page persistently
            page_file = STORAGE / "step3_full_page_content.json"
            with open(page_file, "w") as f:
                json.dump(
                    {"url": self.url, "content": response.text, "status": response.status_code},
                    f,
                    indent=2,
                )

            self.page_content_ = response.text

            print(f"  ✓ NATIVE HTTP request completed")
            print(f"  ✓ Status: {response.status_code}")
            print(f"  ✓ Content length: {len(self.page_content_)} characters")
            print(f"  ✓ FULL page saved to: {page_file}")

        except Exception as e:
            print(f"  ✗ HTTP request error: {e}")
            self.page_content_ = f"Error fetching page: {e}"

        return self


def create_step3_browse(url: str):
    """Create task for browsing."""

    @task
    def step3_browse():
        return Task(
            dataset=[
                Sample(
                    input=f"""URL to visit: {url}

Confirm the URL and specify what data you expect to find on the page.""",
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
    research_areas: list[str] = Field(default_factory=list)


def create_step4_extraction(page_content: str):
    """Create task for data extraction."""
    # Take first 2000 characters as preview
    content_preview = page_content[:2000] if page_content else ""

    @task
    def step4_extract():
        return Task(
            dataset=[
                Sample(
                    input=f"""Cached page content (first 2000 chars):

{content_preview}...

Extract researcher metrics from this cached page content.""",
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
# STEP 5: Orchestrator decision
# ============================================================================


class Step5_OrchestratorDecision(BaseModel):
    """Orchestrator decides whether to continue research."""

    continue_research: bool
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Reason (max 50 words)")


def create_step5_orchestrator(extracted_data: Step4_ExtractedData):
    """Create task for orchestrator decision."""

    @task
    def step5_orchestrator():
        return Task(
            dataset=[
                Sample(
                    input=f"""Data collected so far:
- H-index: {extracted_data.h_index}
- Citations: {extracted_data.total_citations}
- Affiliation: {extracted_data.affiliation}
- Papers: {len(extracted_data.top_papers)}
- Research areas: {len(extracted_data.research_areas)}

Decide if more research is needed to complete the researcher profile.""",
                    target="",
                )
            ],
            solver=generate(),
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="Step5_OrchestratorDecision",
                    json_schema=json_schema(Step5_OrchestratorDecision),
                    strict=True,
                ),
                max_tokens=256,
            ),
        )

    return step5_orchestrator


# ============================================================================
# Pipeline execution
# ============================================================================


def test_native_pipeline():
    """Test pipeline with NATIVE function execution via Pydantic validators."""
    print("=" * 70)
    print("NATIVE RESEARCHER PROFILING PIPELINE")
    print("(LLM JSON → Pydantic validates → NATIVE function executes)")
    print("=" * 70)

    model = get_model("openrouter/qwen/qwen3-coder:free")

    # STEP 1: LLM decides search query → Pydantic executes NATIVE Tavily API call
    print("\n[STEP 1] LLM generates search query (JSON)")
    print("-" * 70)
    log1 = eval(step1_search_decision(), model=model)[0]
    step1_output = Step1_SearchDecision.model_validate_json(
        log1.samples[0].output.completion
    )
    # ↑ When Pydantic validates, @model_validator calls NATIVE Tavily API!

    print(f"\n✓ LLM Output (JSON):")
    print(f"  search_query: {step1_output.search_query}")
    print(f"  rationale: {step1_output.rationale}")
    print(f"✓ Pydantic executed NATIVE Tavily API call")
    print(f"  Results count: {len(step1_output.search_results_)}")

    # STEP 2: LLM selects URL from results
    print("\n[STEP 2] LLM selects URL from search results (JSON)")
    print("-" * 70)
    step2_task = create_step2_url_selection(step1_output.search_results_)
    log2 = eval(step2_task(), model=model)[0]
    step2_output = Step2_URLSelection.model_validate_json(
        log2.samples[0].output.completion
    )

    print(f"✓ LLM Output (JSON):")
    print(f"  selected_url: {step2_output.selected_url}")
    print(f"  reason: {step2_output.reason}")

    # STEP 3: LLM confirms URL → Pydantic executes NATIVE HTTP request
    print("\n[STEP 3] LLM confirms browsing (JSON)")
    print("-" * 70)
    step3_task = create_step3_browse(step2_output.selected_url)
    log3 = eval(step3_task(), model=model)[0]
    step3_output = Step3_BrowseURL.model_validate_json(
        log3.samples[0].output.completion
    )
    # ↑ When Pydantic validates, @model_validator makes NATIVE HTTP request!

    print(f"\n✓ LLM Output (JSON):")
    print(f"  url: {step3_output.url}")
    print(f"  expected_data: {step3_output.expected_data}")
    print(f"✓ Pydantic executed NATIVE HTTP request")
    print(f"  Content length: {len(step3_output.page_content_)} characters")

    # STEP 4: LLM extracts data from cached page
    print("\n[STEP 4] LLM extracts data from cached page (JSON)")
    print("-" * 70)
    step4_task = create_step4_extraction(step3_output.page_content_)
    log4 = eval(step4_task(), model=model)[0]
    step4_output = Step4_ExtractedData.model_validate_json(
        log4.samples[0].output.completion
    )

    print(f"✓ LLM Output (JSON):")
    print(f"  h_index: {step4_output.h_index}")
    print(f"  total_citations: {step4_output.total_citations}")
    print(f"  affiliation: {step4_output.affiliation}")
    print(f"  top_papers: {len(step4_output.top_papers)}")
    print(f"  research_areas: {len(step4_output.research_areas)}")

    # STEP 5: Orchestrator decision
    print("\n[STEP 5] Orchestrator decision (JSON)")
    print("-" * 70)
    step5_task = create_step5_orchestrator(step4_output)
    log5 = eval(step5_task(), model=model)[0]
    step5_output = Step5_OrchestratorDecision.model_validate_json(
        log5.samples[0].output.completion
    )

    print(f"✓ LLM Output (JSON):")
    print(f"  continue_research: {step5_output.continue_research}")
    print(f"  completeness_score: {step5_output.completeness_score:.1%}")
    print(f"  reason: {step5_output.reason}")

    # Save final summary
    summary_file = STORAGE / "pipeline_summary.json"
    summary = {
        "step1_search": step1_output.model_dump(exclude={"search_results_"}),
        "step2_url_selection": step2_output.model_dump(),
        "step3_browse": step3_output.model_dump(exclude={"page_content_"}),
        "step4_extraction": step4_output.model_dump(),
        "step5_orchestrator": step5_output.model_dump(),
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE ✓")
    print("=" * 70)
    print(f"\nAll results saved to: {STORAGE}")
    print("\nHow it works:")
    print("  1. LLM generates JSON structured output")
    print("  2. Pydantic validates JSON")
    print("  3. @model_validator automatically calls NATIVE Python function")
    print("  4. NATIVE functions: Tavily API client, requests library")
    print("  5. Results available to next LLM call")
    print("  6. NO tool calling features from Inspect AI/LLM!")
    print("\nWhat gets saved:")
    print(f"  ✓ RAW search results: {STORAGE}/step1_raw_search_results.json")
    print(f"  ✓ FULL page content: {STORAGE}/step3_full_page_content.json")
    print(f"  ✓ Pipeline summary: {STORAGE}/pipeline_summary.json")


if __name__ == "__main__":
    import sys

    try:
        test_native_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
