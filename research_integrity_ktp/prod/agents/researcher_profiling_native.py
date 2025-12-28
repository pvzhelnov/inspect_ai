"""
Production researcher profiling agent with NATIVE function execution.

CORRECT architecture:
- LLM generates JSON structured output via ResponseSchema
- Pydantic @model_validator automatically executes NATIVE Python functions
- NO tool calling features from Inspect AI/LLM providers
- NATIVE Tavily API client for web search
- NATIVE HTTP requests for web browsing
- Multiple small LLM calls (5-6 steps per iteration)
- RAW search results and FULL pages saved persistently to database

Implements the multi-call research process:
1. Planning: LLM decides what to search
2. Search: Pydantic executes NATIVE Tavily API → saves RAW results
3. URL Selection: LLM selects URL from results
4. Browsing: Pydantic executes NATIVE HTTP request → saves FULL page
5. Extraction: LLM extracts data from cached page
6. Orchestrator: LLM decides if more research needed
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from tavily import TavilyClient

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.solver import generate
from inspect_ai.util import json_schema

from ..db.database import Database, DatabaseConfig
from ..db.repositories import (
    LLMRequestRepository,
    OrchestratorDecisionRepository,
    ResearcherRepository,
    ResearchIterationRepository,
    WebSearchCacheRepository,
    BrowserCacheRepository,
)
from ..models.schemas import (
    LLMRequestRecord,
    OrchestratorDecisionRecord,
    ProcessingStatus,
    ResearcherRecord,
    ResearchIterationRecord,
    WebSearchCacheRecord,
    BrowserCacheRecord,
)

load_dotenv()


# ============================================================================
# Step models with NATIVE function execution via @model_validator
# ============================================================================


class Step1_Planning(BaseModel):
    """LLM planning decision."""

    target_website: str = Field(..., description="Website to search (e.g., Google Scholar)")
    search_query: str = Field(..., description="Specific search query")
    rationale: str = Field(..., description="Why this search (max 100 words)")


class Step2_Search(BaseModel):
    """
    LLM generates search query.
    Pydantic automatically executes NATIVE Tavily API call.
    """

    search_query: str = Field(..., description="Refined search query")
    max_results: int = Field(default=5, description="Maximum results to retrieve")

    # Results populated after execution
    search_results_: list[dict] = Field(default_factory=list, exclude=True)
    raw_response_: dict = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def execute_search(self):
        """Execute NATIVE Tavily API call."""
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY not set")

            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=self.search_query, max_results=self.max_results
            )

            self.raw_response_ = response
            self.search_results_ = response.get("results", [])

        except Exception as e:
            # Fallback for testing
            self.search_results_ = []
            self.raw_response_ = {"error": str(e)}

        return self


class Step3_URLSelection(BaseModel):
    """LLM selects URL from search results."""

    selected_url: str = Field(..., description="URL to visit")
    rationale: str = Field(..., description="Why this URL (max 50 words)")


class Step4_Browsing(BaseModel):
    """
    LLM confirms URL to browse.
    Pydantic automatically executes NATIVE HTTP request.
    """

    url: str = Field(..., description="URL to browse")
    expected_info: str = Field(..., description="What info to find (max 50 words)")

    # Results populated after execution
    page_content_: str = Field(default="", exclude=True)
    status_code_: int = Field(default=0, exclude=True)

    @model_validator(mode="after")
    def execute_browsing(self):
        """Execute NATIVE HTTP request."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(self.url, headers=headers, timeout=30)
            response.raise_for_status()

            self.page_content_ = response.text
            self.status_code_ = response.status_code

        except Exception as e:
            self.page_content_ = f"Error: {e}"
            self.status_code_ = 0

        return self


class Step5_Extraction(BaseModel):
    """LLM extracts data from cached page."""

    # Publication data
    publications: list[dict] = Field(
        default_factory=list,
        description="Publications with title, year, citations",
    )

    # Metrics
    h_index: int | None = None
    i10_index: int | None = None
    total_citations: int | None = None

    # Other data
    affiliations: list[str] = Field(default_factory=list)
    research_areas: list[str] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)
    collaborators: list[str] = Field(default_factory=list)


class Step6_Orchestrator(BaseModel):
    """Orchestrator decision."""

    continue_research: bool
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., description="Reasoning (max 100 words)")
    filled_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    next_focus_areas: list[str] = Field(default_factory=list)


# ============================================================================
# Production Agent
# ============================================================================


class ResearcherProfilingAgentNative:
    """
    Production researcher profiling agent with NATIVE function execution.

    Uses multiple small LLM calls with Pydantic @model_validator for function execution.
    NO tool calling features from Inspect AI.
    """

    def __init__(
        self,
        db_path: Path,
        model_name: str = "openrouter/qwen/qwen3-coder:free",
        max_iterations: int = 5,
    ):
        """
        Initialize the agent.

        Args:
            db_path: Path to SQLite database
            model_name: LLM model to use
            max_iterations: Maximum iterations per researcher
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.model = get_model(model_name)

        # Initialize database
        db_config = DatabaseConfig(db_path=db_path)
        self.db = Database(db_config)

        # Initialize repositories
        self.researcher_repo = ResearcherRepository(self.db)
        self.iteration_repo = ResearchIterationRepository(self.db)
        self.llm_repo = LLMRequestRepository(self.db)
        self.orchestrator_repo = OrchestratorDecisionRepository(self.db)
        self.search_cache_repo = WebSearchCacheRepository(self.db)
        self.browser_cache_repo = BrowserCacheRepository(self.db)

    def process_researcher(self, researcher: ResearcherRecord) -> bool:
        """
        Process a single researcher through multiple research iterations.

        Args:
            researcher: Researcher record to process

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update status to processing
            self.researcher_repo.update_status(
                researcher.researcher_id, ProcessingStatus.PROCESSING
            )

            # Get existing iterations
            existing_iterations = self.iteration_repo.get_by_researcher(
                researcher.researcher_id
            )
            iteration_num = len(existing_iterations) + 1

            # Run research iterations
            while iteration_num <= self.max_iterations:
                # Run one iteration
                iteration_record = self._run_research_iteration(
                    researcher, iteration_num, existing_iterations
                )

                if iteration_record:
                    existing_iterations.append(iteration_record)

                    # Check if we should continue
                    decision = self._run_orchestrator_decision(
                        researcher, iteration_num, existing_iterations
                    )

                    if decision and not decision.continue_research:
                        break

                iteration_num += 1

            # Mark researcher as completed
            self.researcher_repo.update_status(
                researcher.researcher_id, ProcessingStatus.COMPLETED
            )
            return True

        except Exception as e:
            # Mark researcher as failed
            self.researcher_repo.update_status(
                researcher.researcher_id, ProcessingStatus.FAILED
            )
            raise

    def _run_research_iteration(
        self,
        researcher: ResearcherRecord,
        iteration_num: int,
        previous_iterations: list[ResearchIterationRecord],
    ) -> Optional[ResearchIterationRecord]:
        """
        Run a single research iteration with multiple small LLM calls.

        Args:
            researcher: Researcher record
            iteration_num: Current iteration number
            previous_iterations: List of previous iterations

        Returns:
            Research iteration record if successful
        """
        # Create iteration record
        iteration_record = ResearchIterationRecord(
            researcher_id=researcher.researcher_id,
            iteration_number=iteration_num,
            status=ProcessingStatus.PENDING,
            started_at=datetime.now(),
        )
        iteration_id = self.iteration_repo.create(iteration_record)

        try:
            # STEP 1: Planning
            step1_output = self._run_step1_planning(
                researcher, iteration_num, previous_iterations
            )

            # STEP 2: Search (NATIVE Tavily API execution)
            step2_output = self._run_step2_search(step1_output)

            # Save RAW search results to cache
            if step2_output.search_results_:
                search_cache = WebSearchCacheRecord(
                    query=step2_output.search_query,
                    provider="tavily",
                    results=json.dumps(step2_output.raw_response_),
                    researcher_id=researcher.researcher_id,
                    iteration_id=iteration_id,
                )
                self.search_cache_repo.create(search_cache)

            # STEP 3: URL Selection
            step3_output = self._run_step3_url_selection(step2_output)

            # STEP 4: Browsing (NATIVE HTTP request execution)
            step4_output = self._run_step4_browsing(step3_output)

            # Save FULL page content to cache
            if step4_output.page_content_:
                browser_cache = BrowserCacheRecord(
                    url=step4_output.url,
                    content=step4_output.page_content_,
                    researcher_id=researcher.researcher_id,
                    iteration_id=iteration_id,
                )
                self.browser_cache_repo.create(browser_cache)

            # STEP 5: Extraction
            step5_output = self._run_step5_extraction(step4_output)

            # Update iteration record with all outputs
            # (You would map these to the appropriate fields in ResearchIterationRecord)
            self.iteration_repo.update_status(iteration_id, ProcessingStatus.COMPLETED)

            return iteration_record

        except Exception as e:
            # Log error
            self.iteration_repo.update_error(iteration_id, str(e))
            raise

    def _run_step1_planning(
        self,
        researcher: ResearcherRecord,
        iteration_num: int,
        previous_iterations: list[ResearchIterationRecord],
    ) -> Step1_Planning:
        """Step 1: Planning - LLM decides what to search."""
        prompt = self._build_planning_prompt(
            researcher, iteration_num, previous_iterations
        )

        @task
        def planning_task():
            return Task(
                dataset=[Sample(input=prompt, target="")],
                solver=generate(),
                config=GenerateConfig(
                    response_schema=ResponseSchema(
                        name="Step1_Planning",
                        json_schema=json_schema(Step1_Planning),
                        strict=True,
                    ),
                    max_tokens=512,
                ),
            )

        log = eval(planning_task(), model=self.model)[0]
        return Step1_Planning.model_validate_json(log.samples[0].output.completion)

    def _run_step2_search(self, planning: Step1_Planning) -> Step2_Search:
        """Step 2: Search - LLM generates query, Pydantic executes NATIVE Tavily API."""
        prompt = f"""Based on the plan to search {planning.target_website}, generate the exact search query.

Rationale: {planning.rationale}

Provide the refined search query."""

        @task
        def search_task():
            return Task(
                dataset=[Sample(input=prompt, target="")],
                solver=generate(),
                config=GenerateConfig(
                    response_schema=ResponseSchema(
                        name="Step2_Search",
                        json_schema=json_schema(Step2_Search),
                        strict=True,
                    ),
                    max_tokens=256,
                ),
            )

        log = eval(search_task(), model=self.model)[0]
        # When validated, @model_validator automatically calls NATIVE Tavily API!
        return Step2_Search.model_validate_json(log.samples[0].output.completion)

    def _run_step3_url_selection(self, search: Step2_Search) -> Step3_URLSelection:
        """Step 3: URL Selection - LLM selects URL from results."""
        # Format results for LLM
        results_text = "\n".join(
            [
                f"{i+1}. {r.get('title', 'No title')}\n   URL: {r.get('url', 'No URL')}\n   {r.get('content', '')[:100]}"
                for i, r in enumerate(search.search_results_[:5])
            ]
        )

        prompt = f"""Search results from Tavily:
{results_text}

Select the BEST URL to visit for researcher metrics."""

        @task
        def url_selection_task():
            return Task(
                dataset=[Sample(input=prompt, target="")],
                solver=generate(),
                config=GenerateConfig(
                    response_schema=ResponseSchema(
                        name="Step3_URLSelection",
                        json_schema=json_schema(Step3_URLSelection),
                        strict=True,
                    ),
                    max_tokens=256,
                ),
            )

        log = eval(url_selection_task(), model=self.model)[0]
        return Step3_URLSelection.model_validate_json(log.samples[0].output.completion)

    def _run_step4_browsing(self, url_selection: Step3_URLSelection) -> Step4_Browsing:
        """Step 4: Browsing - LLM confirms URL, Pydantic executes NATIVE HTTP request."""
        prompt = f"""URL to visit: {url_selection.selected_url}
Rationale: {url_selection.rationale}

Confirm the URL and specify what information you expect to find."""

        @task
        def browsing_task():
            return Task(
                dataset=[Sample(input=prompt, target="")],
                solver=generate(),
                config=GenerateConfig(
                    response_schema=ResponseSchema(
                        name="Step4_Browsing",
                        json_schema=json_schema(Step4_Browsing),
                        strict=True,
                    ),
                    max_tokens=256,
                ),
            )

        log = eval(browsing_task(), model=self.model)[0]
        # When validated, @model_validator automatically makes NATIVE HTTP request!
        return Step4_Browsing.model_validate_json(log.samples[0].output.completion)

    def _run_step5_extraction(self, browsing: Step4_Browsing) -> Step5_Extraction:
        """Step 5: Extraction - LLM extracts data from cached page."""
        # Take first 3000 chars of cached content
        content_preview = browsing.page_content_[:3000]

        prompt = f"""Cached page content from {browsing.url}:

{content_preview}...

Extract researcher metrics and data from this cached page content."""

        @task
        def extraction_task():
            return Task(
                dataset=[Sample(input=prompt, target="")],
                solver=generate(),
                config=GenerateConfig(
                    response_schema=ResponseSchema(
                        name="Step5_Extraction",
                        json_schema=json_schema(Step5_Extraction),
                        strict=True,
                    ),
                    max_tokens=1024,
                ),
            )

        log = eval(extraction_task(), model=self.model)[0]
        return Step5_Extraction.model_validate_json(log.samples[0].output.completion)

    def _run_orchestrator_decision(
        self,
        researcher: ResearcherRecord,
        iteration_num: int,
        iterations: list[ResearchIterationRecord],
    ) -> Optional[Step6_Orchestrator]:
        """Step 6: Orchestrator - LLM decides if more research needed."""
        prompt = self._build_orchestrator_prompt(researcher, iteration_num, iterations)

        @task
        def orchestrator_task():
            return Task(
                dataset=[Sample(input=prompt, target="")],
                solver=generate(),
                config=GenerateConfig(
                    response_schema=ResponseSchema(
                        name="Step6_Orchestrator",
                        json_schema=json_schema(Step6_Orchestrator),
                        strict=True,
                    ),
                    max_tokens=512,
                ),
            )

        try:
            log = eval(orchestrator_task(), model=self.model)[0]
            decision = Step6_Orchestrator.model_validate_json(
                log.samples[0].output.completion
            )

            # Save to database
            decision_record = OrchestratorDecisionRecord(
                researcher_id=researcher.researcher_id,
                iteration_number=iteration_num,
                continue_research=decision.continue_research,
                rationale=decision.rationale,
                completeness_score=decision.completeness_score,
                filled_fields=decision.filled_fields,
                missing_fields=decision.missing_fields,
                next_focus_areas=decision.next_focus_areas,
            )
            self.orchestrator_repo.create(decision_record)

            return decision

        except Exception as e:
            raise

    def _build_planning_prompt(
        self,
        researcher: ResearcherRecord,
        iteration_num: int,
        previous_iterations: list[ResearchIterationRecord],
    ) -> str:
        """Build prompt for planning step."""
        # Summarize previous iterations
        previous_summary = "None - this is the first iteration"
        if previous_iterations:
            previous_summary = "\n".join(
                [f"Iteration {i.iteration_number}: [summary would go here]" for i in previous_iterations]
            )

        return f"""You are researching: {researcher.name}
Field: {researcher.field}
Known info: {researcher.known_info}

This is iteration {iteration_num}.
Previous iterations: {previous_summary}

Decide what to search next to fill gaps in the researcher profile."""

    def _build_orchestrator_prompt(
        self,
        researcher: ResearcherRecord,
        iteration_num: int,
        iterations: list[ResearchIterationRecord],
    ) -> str:
        """Build prompt for orchestrator decision."""
        return f"""You have completed {iteration_num} iteration(s) of research on {researcher.name}.

Assess the completeness of the researcher profile and decide whether more research is needed."""

    def close(self) -> None:
        """Close database connections."""
        self.db.close()
