#!/usr/bin/env python3
"""
Live test for researcher profiling agent with actual OpenRouter API calls.
Logs all outputs to files for review.

Run from repo root:
OPENROUTER_API_KEY=<key> uv run python test_researcher_profiling_live.py
"""

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate
from inspect_ai.util import json_schema


# Define schemas (copied from test file to avoid pytest dependency)
class ResearcherReflection(BaseModel):
    """Reflection on the researcher's current data state."""

    current_knowledge_summary: str = Field(
        ..., description="Summary of what is currently known about the researcher"
    )
    identified_gaps: list[str] = Field(
        ..., description="List of information gaps that need to be filled"
    )
    priority_areas: list[str] = Field(
        ..., description="Priority areas for next research iteration"
    )


class PreviousIterationReview(BaseModel):
    """Review of previous research iterations."""

    iterations_completed: int = Field(
        ..., description="Number of iterations completed so far"
    )
    sources_consulted: list[str] = Field(
        ...,
        description="List of sources/places already consulted in previous iterations",
    )
    information_collected: dict[str, str] = Field(
        ..., description="Summary of key information collected from each source"
    )


class SearchLocationSelection(BaseModel):
    """Selection of where to search next."""

    selected_location: str = Field(
        ...,
        description="The website or database to search (e.g., 'Google Scholar', 'ResearchGate', 'University homepage')",
    )
    rationale: str = Field(
        ...,
        description="Why this location was chosen and what information we expect to find",
    )
    avoids_overlap: bool = Field(
        ...,
        description="Confirms this location has not been sufficiently covered in previous iterations",
    )


class LanguageSelection(BaseModel):
    """Selection of search language."""

    language: str = Field(
        ..., description="Language code (e.g., 'en', 'de', 'fr', 'zh', 'ja')"
    )
    language_name: str = Field(..., description="Full language name (e.g., 'English')")
    rationale: str = Field(
        ..., description="Why this language is most appropriate for this search"
    )


class WebSearchResults(BaseModel):
    """Web search results for finding the target URL."""

    search_query: str = Field(..., description="The search query used")
    target_url: str = Field(
        ..., description="The URL identified as most relevant for research"
    )
    alternative_urls: list[str] = Field(
        default_factory=list, description="Alternative URLs that might be useful"
    )
    search_summary: str = Field(
        ..., description="Summary of what was found in the search results"
    )


class SearchStrategy(BaseModel):
    """Detailed strategy for browsing and extracting data."""

    website_structure: str = Field(
        ..., description="Expected structure of the target website"
    )
    navigation_plan: list[str] = Field(
        ..., description="Step-by-step plan for navigating the website"
    )
    data_extraction_points: list[str] = Field(
        ..., description="Specific data points to extract and where to find them"
    )
    search_queries_to_use: list[str] = Field(
        ..., description="Specific search queries or keywords to use on the site"
    )


class BrowsingResults(BaseModel):
    """Results from browsing the website."""

    pages_visited: list[str] = Field(
        ..., description="List of page URLs visited during browsing"
    )
    page_summaries: dict[str, str] = Field(
        ..., description="Summary of content from each page visited"
    )
    full_page_cache_status: str = Field(
        ...,
        description="Confirmation that full pages were saved to the Inspect AI log",
    )


class ExtractedData(BaseModel):
    """Data extracted from the browsed pages."""

    publications: list[str] = Field(
        default_factory=list, description="Key publications by the researcher"
    )
    citations: dict[str, int] = Field(
        default_factory=dict, description="Citation counts from various sources"
    )
    affiliations: list[str] = Field(
        default_factory=list, description="Current and past affiliations"
    )
    research_areas: list[str] = Field(
        default_factory=list, description="Primary research areas"
    )
    collaborators: list[str] = Field(
        default_factory=list, description="Frequent collaborators"
    )
    awards: list[str] = Field(
        default_factory=list, description="Awards and honors received"
    )
    h_index: int | None = Field(default=None, description="H-index if available")
    additional_info: dict[str, str] = Field(
        default_factory=dict, description="Any other relevant information"
    )


class ResearchIterationOutput(BaseModel):
    """Complete output from one research iteration."""

    step_1_reflection: ResearcherReflection = Field(
        ..., description="Reflection on current researcher data"
    )
    step_2_previous_review: PreviousIterationReview = Field(
        ..., description="Review of previous iterations"
    )
    step_3_location_selection: SearchLocationSelection = Field(
        ..., description="Selection of search location"
    )
    step_4_language_selection: LanguageSelection = Field(
        ..., description="Selection of search language"
    )
    step_5_web_search: WebSearchResults = Field(
        ..., description="Web search results with full results logged"
    )
    step_6_search_strategy: SearchStrategy = Field(
        ..., description="Strategy for browsing and data extraction"
    )
    step_7_browsing: BrowsingResults = Field(
        ..., description="Browsing results with full pages logged"
    )
    step_8_extracted_data: ExtractedData = Field(
        ..., description="Data extracted from cached pages"
    )


class ProfileCompletenessAssessment(BaseModel):
    """Assessment of researcher profile completeness."""

    completeness_score: float = Field(
        ..., description="Completeness score from 0.0 to 1.0", ge=0.0, le=1.0
    )
    filled_fields: list[str] = Field(
        ..., description="List of profile fields that are well-filled"
    )
    missing_fields: list[str] = Field(
        ..., description="List of profile fields that are missing or incomplete"
    )
    data_quality_notes: str = Field(
        ..., description="Notes on the quality and reliability of collected data"
    )


class OrchestratorDecision(BaseModel):
    """Decision from the orchestrator agent."""

    continue_research: bool = Field(
        ..., description="Whether to continue with another research iteration"
    )
    rationale: str = Field(..., description="Rationale for the decision")
    assessment: ProfileCompletenessAssessment = Field(
        ..., description="Assessment of profile completeness"
    )
    next_focus_areas: list[str] = Field(
        default_factory=list,
        description="Suggested focus areas for next iteration if continuing",
    )


# Scorers
@scorer(metrics=[accuracy(), stderr()])
def score_research_iteration():
    """Score a research iteration output."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            from pydantic import ValidationError

            iteration_output = ResearchIterationOutput.model_validate_json(
                state.output.completion
            )

            checks = [
                iteration_output.step_1_reflection.current_knowledge_summary != "",
                len(iteration_output.step_2_previous_review.sources_consulted) >= 0,
                iteration_output.step_3_location_selection.selected_location != "",
                iteration_output.step_4_language_selection.language != "",
                iteration_output.step_5_web_search.target_url != "",
                len(iteration_output.step_6_search_strategy.navigation_plan) > 0,
                len(iteration_output.step_7_browsing.pages_visited) > 0,
                (
                    len(iteration_output.step_8_extracted_data.publications) > 0
                    or len(iteration_output.step_8_extracted_data.affiliations) > 0
                    or len(iteration_output.step_8_extracted_data.research_areas) > 0
                ),
            ]

            value = CORRECT if all(checks) else INCORRECT

            return Score(
                value=value,
                answer=state.output.completion,
                explanation=f"Completed {sum(checks)}/{len(checks)} validation checks",
            )

        except ValidationError as ex:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Error parsing response: {ex}",
            )

    return score


@scorer(metrics=[accuracy(), stderr()])
def score_orchestrator_decision():
    """Score an orchestrator decision."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            from pydantic import ValidationError

            decision = OrchestratorDecision.model_validate_json(
                state.output.completion
            )

            checks = [
                decision.rationale != "",
                0.0 <= decision.assessment.completeness_score <= 1.0,
                len(decision.assessment.filled_fields) >= 0,
                len(decision.assessment.missing_fields) >= 0,
                decision.assessment.data_quality_notes != "",
            ]

            value = CORRECT if all(checks) else INCORRECT

            return Score(
                value=value,
                answer=state.output.completion,
                explanation=f"Completed {sum(checks)}/{len(checks)} validation checks. Completeness: {decision.assessment.completeness_score:.2%}",
            )

        except ValidationError as ex:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Error parsing response: {ex}",
            )

    return score


# Tasks
@task
def research_agent_single_iteration():
    """Single iteration of the research agent."""
    return Task(
        dataset=[
            Sample(
                input="""You are a research profiling agent. Research Dr. Geoffrey Hinton, a highly cited AI researcher.

This is iteration 1. Previous iterations: none.

Current known information about the researcher:
- Name: Geoffrey Hinton
- Field: Artificial Intelligence, Deep Learning
- Known for: Backpropagation, Deep Learning

Your task is to perform ONE research iteration following these steps:
1. Reflect on the current researcher data
2. Review results of previous iterations (none in this case)
3. Select a place to search (not overlapping with previous results)
4. Select the most appropriate language for the search
5. Use web search to find the correct URL for the selected place
6. Formulate a detailed search strategy for this website
7. Use the information to simulate browsing (describe what pages you would visit)
8. Extract relevant data about the researcher

Provide your response as a structured JSON output following the ResearchIterationOutput schema.""",
                target="",
            )
        ],
        solver=generate(),
        scorer=score_research_iteration(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="ResearchIterationOutput",
                json_schema=json_schema(ResearchIterationOutput),
                description=ResearchIterationOutput.__doc__,
                strict=True,
            ),
            max_tokens=8192,
        ),
    )


@task
def orchestrator_agent_task():
    """Orchestrator agent decision-making task."""
    return Task(
        dataset=[
            Sample(
                input="""You are an orchestrator agent managing a researcher profiling system.

You have completed 2 iterations of research on Dr. Geoffrey Hinton:

Iteration 1: Collected basic biographical data from Wikipedia
- Publications: ~200+ papers
- H-index: 160+
- Affiliations: University of Toronto, Google Brain
- Research areas: Deep Learning, Neural Networks

Iteration 2: Collected citation data from Google Scholar
- Total citations: 500,000+
- Key papers: "Deep Learning" (2015), "Reducing the Dimensionality of Data with Neural Networks" (2006)
- Awards: Turing Award (2018)

Assess the completeness of the researcher profile and decide whether to continue research or if the dataset is sufficiently filled.

Provide your response as a structured JSON output following the OrchestratorDecision schema.""",
                target="",
            )
        ],
        solver=generate(),
        scorer=score_orchestrator_decision(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="OrchestratorDecision",
                json_schema=json_schema(OrchestratorDecision),
                description=OrchestratorDecision.__doc__,
                strict=True,
            ),
            max_tokens=4096,
        ),
    )


def save_output_to_file(log, test_name: str):
    """Save test output to a file for review."""
    output_dir = Path("logs/researcher_profiling_live")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the full log
    log_file = output_dir / f"{test_name}_log.json"
    with open(log_file, "w") as f:
        # Extract relevant info from log
        log_data = {
            "status": log.status,
            "samples": [
                {
                    "input": sample.input,
                    "output": sample.output.completion if sample.output else None,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": str(msg.content)[:500]
                            + ("..." if len(str(msg.content)) > 500 else ""),
                        }
                        for msg in sample.messages[-5:]  # Last 5 messages
                    ],
                }
                for sample in log.samples
            ],
            "results": {
                "scores": [
                    {
                        "name": score.name,
                        "metrics": {
                            name: {
                                "value": metric.value,
                            }
                            for name, metric in score.metrics.items()
                        },
                    }
                    for score in log.results.scores
                ]
            },
        }
        json.dump(log_data, f, indent=2)

    # Save the completion output separately
    if log.samples and log.samples[0].output:
        completion_file = output_dir / f"{test_name}_completion.json"
        with open(completion_file, "w") as f:
            f.write(log.samples[0].output.completion)

    print(f"   📁 Saved outputs to: {output_dir}")
    print(f"      - {log_file.name}")
    print(f"      - {completion_file.name}")
    return output_dir


def test_research_agent_live():
    """Test research agent with actual OpenRouter API."""
    print("\n" + "=" * 70)
    print("TEST 1: Research Agent with OpenRouter (Qwen 4B - Free)")
    print("=" * 70)

    model = get_model(
        "openrouter/qwen/qwen3-4b:free",  # Use smaller 4B model - faster and more reliable
        config=GenerateConfig(
            max_tokens=4096,  # Reduced for smaller model
            timeout=60,  # 60 second timeout
        ),
    )

    print(
        "\n🚀 Running research agent with Qwen 4B (Free) via OpenRouter..."
    )
    print("   Model will generate structured ResearchIterationOutput")
    print("   Timeout: 60 seconds")

    log = eval(
        research_agent_single_iteration(),
        model=model,
        model_args={"provider": {"require_parameters": True}},
    )[0]

    # Save outputs
    output_dir = save_output_to_file(log, "research_agent")

    # Display results
    print(f"\n📊 Results:")
    print(f"   - Status: {log.status}")
    print(f"   - Accuracy: {log.results.scores[0].metrics['accuracy'].value}")
    print(
        f"   - Stderr: {log.results.scores[0].metrics['stderr'].value}"
    )

    # Parse and display the structured output
    if log.samples and log.samples[0].output:
        try:
            output = ResearchIterationOutput.model_validate_json(
                log.samples[0].output.completion
            )
            print(f"\n✅ Structured Output Validation: PASSED")
            print(f"\n📝 Research Iteration Summary:")
            print(
                f"   - Selected Location: {output.step_3_location_selection.selected_location}"
            )
            print(
                f"   - Language: {output.step_4_language_selection.language_name}"
            )
            print(
                f"   - Target URL: {output.step_5_web_search.target_url}"
            )
            print(
                f"   - Pages Visited: {len(output.step_7_browsing.pages_visited)}"
            )
            print(
                f"   - Publications Found: {len(output.step_8_extracted_data.publications)}"
            )
            print(
                f"   - Affiliations: {', '.join(output.step_8_extracted_data.affiliations[:3])}"
            )
            if output.step_8_extracted_data.h_index:
                print(
                    f"   - H-index: {output.step_8_extracted_data.h_index}"
                )
        except Exception as e:
            print(f"\n❌ Structured Output Validation: FAILED")
            print(f"   Error: {e}")

    assert log.status == "success", f"Expected success but got {log.status}"
    assert (
        log.results.scores[0].metrics["accuracy"].value > 0
    ), "Expected accuracy > 0"

    return log


def test_orchestrator_agent_live():
    """Test orchestrator agent with actual OpenRouter API."""
    print("\n" + "=" * 70)
    print("TEST 2: Orchestrator Agent with OpenRouter (Qwen 4B - Free)")
    print("=" * 70)

    model = get_model(
        "openrouter/qwen/qwen3-4b:free",
        config=GenerateConfig(
            max_tokens=2048,  # Reduced for smaller model
            timeout=30,  # 30 second timeout
        ),
    )

    print(
        "\n🚀 Running orchestrator agent with Qwen 4B (Free) via OpenRouter..."
    )
    print("   Model will make decision about continuing research")
    print("   Timeout: 30 seconds")

    log = eval(
        orchestrator_agent_task(),
        model=model,
        model_args={"provider": {"require_parameters": True}},
    )[0]

    # Save outputs
    output_dir = save_output_to_file(log, "orchestrator_agent")

    # Display results
    print(f"\n📊 Results:")
    print(f"   - Status: {log.status}")
    print(f"   - Accuracy: {log.results.scores[0].metrics['accuracy'].value}")
    print(
        f"   - Stderr: {log.results.scores[0].metrics['stderr'].value}"
    )

    # Parse and display the structured output
    if log.samples and log.samples[0].output:
        try:
            output = OrchestratorDecision.model_validate_json(
                log.samples[0].output.completion
            )
            print(f"\n✅ Structured Output Validation: PASSED")
            print(f"\n🎯 Orchestrator Decision:")
            print(
                f"   - Continue Research: {output.continue_research}"
            )
            print(
                f"   - Completeness Score: {output.assessment.completeness_score:.2%}"
            )
            print(
                f"   - Filled Fields: {len(output.assessment.filled_fields)}"
            )
            print(
                f"   - Missing Fields: {len(output.assessment.missing_fields)}"
            )
            print(f"   - Rationale: {output.rationale[:200]}...")
        except Exception as e:
            print(f"\n❌ Structured Output Validation: FAILED")
            print(f"   Error: {e}")

    assert log.status == "success", f"Expected success but got {log.status}"
    assert (
        log.results.scores[0].metrics["accuracy"].value > 0
    ), "Expected accuracy > 0"

    return log


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("RESEARCHER PROFILING AGENT - LIVE TEST WITH OPENROUTER")
        print("=" * 70)
        print("\n🔑 Using OpenRouter API")
        print("📝 All outputs will be logged to files")

        # Run tests
        research_log = test_research_agent_live()
        orchestrator_log = test_orchestrator_agent_live()

        print("\n" + "=" * 70)
        print("ALL LIVE TESTS PASSED! ✅")
        print("=" * 70)
        print("\n📁 Check logs/researcher_profiling_live/ for detailed outputs")

        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
