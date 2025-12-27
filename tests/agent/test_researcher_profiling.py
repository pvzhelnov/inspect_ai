"""
Researcher Profiling Agent System

This module implements a multi-agent system for deep research on highly cited scientists.
The system consists of:
1. Research Agent: Performs structured research iterations with web search and browsing
2. Orchestrator Agent: Manages the research loop and decides when profiling is complete
"""

import pytest
from pydantic import BaseModel, Field, ValidationError
from test_helpers.utils import force_runapi, skip_if_no_openrouter

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
from inspect_ai.solver import TaskState, generate, use_tools
from inspect_ai.tool import web_browser, web_search
from inspect_ai.util import json_schema

# ============================================================================
# Research Agent Schemas
# ============================================================================


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


# ============================================================================
# Orchestrator Agent Schemas
# ============================================================================


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


# ============================================================================
# Validated Model for Ground Truth
# ============================================================================


class ValidatedResearcherProfile(BaseModel):
    """Manually curated ground truth for researcher profile."""

    researcher_name: str
    verified_publications_count: int = Field(
        ..., description="Verified number of publications"
    )
    verified_citation_count: int | None = Field(
        default=None, description="Verified citation count"
    )
    verified_h_index: int | None = Field(default=None, description="Verified h-index")
    verified_affiliations: list[str] = Field(
        default_factory=list, description="Verified affiliations"
    )
    verified_research_areas: list[str] = Field(
        default_factory=list, description="Verified research areas"
    )
    key_achievements: list[str] = Field(
        default_factory=list, description="Key achievements to verify"
    )


# ============================================================================
# Scorers
# ============================================================================


@scorer(metrics=[accuracy(), stderr()])
def score_research_iteration():
    """Score a research iteration output."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
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
            decision = OrchestratorDecision.model_validate_json(state.output.completion)

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


# ============================================================================
# Tasks
# ============================================================================


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
def research_agent_with_tools():
    """Research agent with actual web_search and web_browser tools."""
    return Task(
        dataset=[
            Sample(
                input="""You are a research profiling agent with web search and browsing capabilities.

Research Dr. Yann LeCun, a highly cited AI researcher.

This is iteration 1. Previous iterations: none.

Current known information:
- Name: Yann LeCun
- Field: Computer Science, AI
- Known for: Convolutional Neural Networks

Your task:
1. Use web_search to find information about Dr. Yann LeCun
2. Reflect on the search results
3. Select the most promising URL from the search results
4. Formulate a strategy for what information to extract
5. Document your findings

Note: The web_search tool will automatically log ALL search results to the Inspect AI log.
The web_browser tool will save FULL PAGE content to the log.

After gathering information, provide a structured response following the ResearchIterationOutput schema,
describing what you found and what you would extract from the pages.""",
                target="",
            )
        ],
        solver=[
            use_tools([web_search(providers="google"), *web_browser()]),
            generate(),
        ],
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


# ============================================================================
# Tests
# ============================================================================


def eval_researcher_profiling(task_fn, model):
    """Helper to evaluate researcher profiling tasks."""
    model_args = {"provider": {"require_parameters": True}}
    log = eval(task_fn, model=model, model_args=model_args)[0]
    assert log.status == "success"
    return log


@force_runapi
@skip_if_no_openrouter
@pytest.mark.parametrize(
    "model_name",
    [
        "openrouter/qwen/qwen3-235b-a22b:free",
        "openrouter/anthropic/claude-3.5-sonnet",
    ],
)
def test_research_agent_iteration(model_name: str):
    """Test a single research agent iteration with structured output."""
    model = get_model(
        model_name,
        config=GenerateConfig(
            max_tokens=8192,
        ),
    )
    log = eval_researcher_profiling(research_agent_single_iteration(), model)
    # Check that we got a valid structured response
    assert log.results.scores[0].metrics["accuracy"].value >= 0.0


@force_runapi
@skip_if_no_openrouter
@pytest.mark.parametrize(
    "model_name",
    [
        "openrouter/anthropic/claude-3.5-sonnet",
    ],
)
def test_research_agent_with_tools(model_name: str):
    """Test research agent with actual web_search and web_browser tools."""
    model = get_model(
        model_name,
        config=GenerateConfig(
            max_tokens=8192,
        ),
    )
    log = eval_researcher_profiling(research_agent_with_tools(), model)
    # Check that we got a valid structured response
    assert log.results.scores[0].metrics["accuracy"].value >= 0.0
    # Verify that tool calls were made and logged
    # The log should contain web_search results and web_browser interactions


@force_runapi
@skip_if_no_openrouter
@pytest.mark.parametrize(
    "model_name",
    [
        "openrouter/qwen/qwen3-235b-a22b:free",
        "openrouter/anthropic/claude-3.5-sonnet",
    ],
)
def test_orchestrator_agent(model_name: str):
    """Test the orchestrator agent decision-making with structured output."""
    model = get_model(
        model_name,
        config=GenerateConfig(
            max_tokens=4096,
        ),
    )
    log = eval_researcher_profiling(orchestrator_agent_task(), model)
    # Check that we got a valid structured response
    assert log.results.scores[0].metrics["accuracy"].value >= 0.0


# ============================================================================
# Integration Test with Full Loop
# ============================================================================


@task
def full_researcher_profiling_loop():
    """Full researcher profiling loop with orchestrator managing iterations."""
    return Task(
        dataset=[
            Sample(
                input="""You are orchestrating a researcher profiling system for Dr. Yann LeCun.

Initial information:
- Name: Yann LeCun
- Field: Computer Science, AI
- Known for: Convolutional Neural Networks

Simulate 3 iterations of the research agent, where each iteration:
1. The research agent performs structured research (following the ResearchIterationOutput schema)
2. After each iteration, you (as orchestrator) assess completeness using the OrchestratorDecision schema
3. You decide whether to continue or stop

Describe this process and provide a final OrchestratorDecision indicating the research is complete.

Provide your final decision as structured JSON following the OrchestratorDecision schema.""",
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
            max_tokens=8192,
        ),
    )


@force_runapi
@skip_if_no_openrouter
@pytest.mark.parametrize(
    "model_name",
    [
        "openrouter/qwen/qwen3-235b-a22b:free",
    ],
)
def test_full_profiling_loop(model_name: str):
    """Test the full researcher profiling loop with orchestrator."""
    model = get_model(
        model_name,
        config=GenerateConfig(
            max_tokens=8192,
        ),
    )
    log = eval_researcher_profiling(full_researcher_profiling_loop(), model)
    assert log.results.scores[0].metrics["accuracy"].value >= 0.0
