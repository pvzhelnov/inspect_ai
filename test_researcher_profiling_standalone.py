#!/usr/bin/env python3
"""
Standalone test for researcher profiling agent system using MockLLM.
This script can be run without pytest to verify the implementation.
Run from repo root: python test_researcher_profiling_standalone.py
"""

import json
import sys

from pydantic import BaseModel, Field

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelOutput, ResponseSchema, get_model
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


# Copy schemas from test file
class ResearcherReflection(BaseModel):
    """Reflection on the researcher's current data state."""

    current_knowledge_summary: str = Field(
        ...,
        description="Summary of what is currently known about the researcher",
    )
    identified_gaps: list[str] = Field(
        ...,
        description="List of information gaps that need to be filled",
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
        ...,
        description="Summary of key information collected from each source",
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
        default_factory=list,
        description="Alternative URLs that might be useful",
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
        default_factory=dict,
        description="Citation counts from various sources",
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
        ...,
        description="Completeness score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
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

            # Parse the structured output
            iteration_output = ResearchIterationOutput.model_validate_json(
                state.output.completion
            )

            # Validate all steps were completed
            checks = [
                iteration_output.step_1_reflection.current_knowledge_summary != "",
                len(iteration_output.step_2_previous_review.sources_consulted) >= 0,
                iteration_output.step_3_location_selection.selected_location != "",
                iteration_output.step_4_language_selection.language != "",
                iteration_output.step_5_web_search.target_url != "",
                len(iteration_output.step_6_search_strategy.navigation_plan) > 0,
                len(iteration_output.step_7_browsing.pages_visited) > 0,
                # At least some data should be extracted
                (
                    len(iteration_output.step_8_extracted_data.publications) > 0
                    or len(iteration_output.step_8_extracted_data.affiliations) > 0
                    or len(iteration_output.step_8_extracted_data.research_areas) > 0
                ),
            ]

            # Check against ground truth if provided
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

            # Validate decision is well-formed
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
                target="",  # No specific target for this example
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


def test_research_agent_with_mockllm():
    """Test research agent with MockLLM to verify schema validation."""
    print("Testing Research Agent with MockLLM...")

    # Create a valid ResearchIterationOutput
    mock_output = ResearchIterationOutput(
        step_1_reflection=ResearcherReflection(
            current_knowledge_summary="Dr. Geoffrey Hinton is a pioneer in deep learning",
            identified_gaps=["Citation counts", "Recent publications"],
            priority_areas=["Google Scholar profile", "University homepage"],
        ),
        step_2_previous_review=PreviousIterationReview(
            iterations_completed=0,
            sources_consulted=[],
            information_collected={},
        ),
        step_3_location_selection=SearchLocationSelection(
            selected_location="Google Scholar",
            rationale="Best source for citation metrics and publication lists",
            avoids_overlap=True,
        ),
        step_4_language_selection=LanguageSelection(
            language="en",
            language_name="English",
            rationale="Primary language for academic publications in AI",
        ),
        step_5_web_search=WebSearchResults(
            search_query="Geoffrey Hinton Google Scholar",
            target_url="https://scholar.google.com/citations?user=JicYPdAAAAAJ",
            alternative_urls=[
                "https://en.wikipedia.org/wiki/Geoffrey_Hinton",
                "https://www.cs.toronto.edu/~hinton/",
            ],
            search_summary="Found official Google Scholar profile and university page",
        ),
        step_6_search_strategy=SearchStrategy(
            website_structure="Google Scholar profile with publications list and citation metrics",
            navigation_plan=[
                "Access main profile page",
                "Extract h-index and citation count",
                "Collect top publications",
            ],
            data_extraction_points=[
                "H-index from profile header",
                "Total citations from profile",
                "Publication titles and years",
            ],
            search_queries_to_use=["Geoffrey Hinton", "neural networks"],
        ),
        step_7_browsing=BrowsingResults(
            pages_visited=["https://scholar.google.com/citations?user=JicYPdAAAAAJ"],
            page_summaries={
                "https://scholar.google.com/citations?user=JicYPdAAAAAJ": "Profile page with 500,000+ citations, h-index 168"
            },
            full_page_cache_status="Full page content saved to Inspect AI log",
        ),
        step_8_extracted_data=ExtractedData(
            publications=[
                "Deep Learning (2015)",
                "Reducing the Dimensionality of Data with Neural Networks (2006)",
                "ImageNet Classification with Deep CNNs (2012)",
            ],
            citations={
                "Google Scholar": 500000,
            },
            affiliations=["University of Toronto", "Google Brain", "Vector Institute"],
            research_areas=["Deep Learning", "Neural Networks", "Backpropagation"],
            collaborators=["Yann LeCun", "Yoshua Bengio"],
            awards=["Turing Award 2018"],
            h_index=168,
            additional_info={
                "emeritus_professor": "University of Toronto",
                "notable_work": "Capsule Networks",
            },
        ),
    )

    # Convert to JSON
    mock_json = mock_output.model_dump_json()

    # Create MockLLM model with custom output
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content(model="mockllm", content=mock_json)],
    )

    # Run the task
    log = eval(
        research_agent_single_iteration(),
        model=model,
    )[0]

    assert log.status == "success", f"Task failed with status: {log.status}"
    # The scorer should validate the structured output
    accuracy = log.results.scores[0].metrics["accuracy"].value
    assert (
        accuracy == 1.0
    ), f"Expected accuracy 1.0 but got {accuracy}"

    print("✅ Research Agent test PASSED")
    print(f"   - Task status: {log.status}")
    print(f"   - Accuracy: {accuracy}")
    print(
        f"   - Output length: {len(log.samples[0].output.completion)} chars"
    )


def test_orchestrator_agent_with_mockllm():
    """Test orchestrator agent with MockLLM to verify decision schema."""
    print("\nTesting Orchestrator Agent with MockLLM...")

    # Create a valid OrchestratorDecision
    mock_decision = OrchestratorDecision(
        continue_research=False,
        rationale="Profile is sufficiently complete with comprehensive data from multiple sources",
        assessment=ProfileCompletenessAssessment(
            completeness_score=0.85,
            filled_fields=[
                "publications",
                "citations",
                "h_index",
                "affiliations",
                "research_areas",
                "awards",
            ],
            missing_fields=["grant_funding", "patents"],
            data_quality_notes="High-quality data from authoritative sources (Google Scholar, university page)",
        ),
        next_focus_areas=[],
    )

    # Convert to JSON
    mock_json = mock_decision.model_dump_json()

    # Create MockLLM model
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content(model="mockllm", content=mock_json)],
    )

    # Run the task
    log = eval(
        orchestrator_agent_task(),
        model=model,
    )[0]

    assert log.status == "success", f"Task failed with status: {log.status}"
    # Verify the decision was validated correctly
    accuracy = log.results.scores[0].metrics["accuracy"].value
    assert (
        accuracy == 1.0
    ), f"Expected accuracy 1.0 but got {accuracy}"

    # Parse the output to verify structure
    output = json.loads(log.samples[0].output.completion)
    assert (
        output["continue_research"] is False
    ), f"Expected continue_research=False but got {output['continue_research']}"
    assert (
        output["assessment"]["completeness_score"] == 0.85
    ), f"Expected completeness_score=0.85 but got {output['assessment']['completeness_score']}"

    print("✅ Orchestrator Agent test PASSED")
    print(f"   - Task status: {log.status}")
    print(f"   - Accuracy: {accuracy}")
    print(f"   - Continue research: {output['continue_research']}")
    print(
        f"   - Completeness score: {output['assessment']['completeness_score']}"
    )


def test_research_iteration_validation():
    """Test that invalid schema outputs are rejected."""
    print("\nTesting Schema Validation (invalid output)...")

    # Create an invalid output (missing required fields)
    invalid_json = json.dumps(
        {
            "step_1_reflection": {
                "current_knowledge_summary": "Some text",
                # Missing other required fields
            }
        }
    )

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(model="mockllm", content=invalid_json)
        ],
    )

    log = eval(
        research_agent_single_iteration(),
        model=model,
    )[0]

    # Should fail because of invalid schema
    assert log.status == "success", "Task should complete"
    # ...but the scorer should mark it as incorrect
    accuracy = log.results.scores[0].metrics["accuracy"].value
    assert (
        accuracy == 0.0
    ), f"Expected accuracy 0.0 for invalid schema but got {accuracy}"

    print("✅ Schema Validation test PASSED")
    print(f"   - Task status: {log.status}")
    print(f"   - Accuracy: {accuracy} (correctly rejected invalid schema)")


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("RESEARCHER PROFILING AGENT SYSTEM - STANDALONE TEST")
        print("=" * 70)

        test_research_agent_with_mockllm()
        test_orchestrator_agent_with_mockllm()
        test_research_iteration_validation()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✅")
        print("=" * 70)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
