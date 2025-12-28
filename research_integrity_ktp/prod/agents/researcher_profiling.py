"""
Production researcher profiling agent with database integration.

Implements the 8-step research process with:
- Database logging of all steps
- LLM request tracking
- Web search and browser caching
- Error handling and retry logic
"""

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, use_tools
from inspect_ai.tool import web_browser, web_search
from inspect_ai.util import json_schema

from ..db.database import Database, DatabaseConfig
from ..db.repositories import (
    LLMRequestRepository,
    OrchestratorDecisionRepository,
    ResearcherRepository,
    ResearchIterationRepository,
)
from ..models.schemas import (
    LLMRequestRecord,
    OrchestratorDecision,
    OrchestratorDecisionRecord,
    ProcessingStatus,
    ResearcherRecord,
    ResearchIterationOutput,
    ResearchIterationRecord,
)

load_dotenv()


class ResearcherProfilingAgent:
    """
    Production researcher profiling agent with database integration.

    Handles:
    - Research iterations with 8-step structured output
    - Orchestrator decisions for iteration management
    - Database logging of all requests and responses
    - Web search and browser caching
    - Progress tracking and error recovery
    """

    def __init__(
        self,
        db_path: Path,
        model_name: str = "openrouter/qwen/qwen3-coder:free",
        max_iterations: int = 5,
        enable_tools: bool = True,
    ):
        """
        Initialize the agent.

        Args:
            db_path: Path to SQLite database
            model_name: LLM model to use
            max_iterations: Maximum iterations per researcher
            enable_tools: Enable web_search and web_browser tools
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.enable_tools = enable_tools

        # Initialize database
        db_config = DatabaseConfig(db_path=db_path)
        self.db = Database(db_config)

        # Initialize repositories
        self.researcher_repo = ResearcherRepository(self.db)
        self.iteration_repo = ResearchIterationRepository(self.db)
        self.llm_repo = LLMRequestRepository(self.db)
        self.orchestrator_repo = OrchestratorDecisionRepository(self.db)

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
        Run a single research iteration.

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
            # Create LLM request record
            request_id = str(uuid.uuid4())
            llm_request = LLMRequestRecord(
                request_id=request_id,
                researcher_id=researcher.researcher_id,
                iteration_id=iteration_id,
                request_type="research_iteration",
                model_name=self.model_name,
                prompt=self._build_research_prompt(
                    researcher, iteration_num, previous_iterations
                ),
                status=ProcessingStatus.PENDING,
            )
            self.llm_repo.create(llm_request)

            # Run the task
            start_time = time.time()
            result = self._execute_research_task(llm_request.prompt)
            latency_ms = int((time.time() - start_time) * 1000)

            # Parse the output
            output = ResearchIterationOutput.model_validate_json(
                result.samples[0].output.completion
            )

            # Update LLM request with response
            # Note: Token counts would come from actual model response
            self.llm_repo.update_response(
                request_id=request_id,
                response=result.samples[0].output.completion,
                tokens_prompt=0,  # Would get from model response
                tokens_completion=0,  # Would get from model response
                latency_ms=latency_ms,
                cost_usd=0.0,  # Would calculate based on model pricing
            )

            # Update iteration record with output
            iteration_record.step_1_reflection = output.step_1_reflection
            iteration_record.step_2_previous_review = output.step_2_previous_review
            iteration_record.step_3_location_selection = (
                output.step_3_location_selection
            )
            iteration_record.step_4_language_selection = (
                output.step_4_language_selection
            )
            iteration_record.step_5_web_search = output.step_5_web_search
            iteration_record.step_6_search_strategy = output.step_6_search_strategy
            iteration_record.step_7_browsing = output.step_7_browsing
            iteration_record.step_8_extracted_data = output.step_8_extracted_data

            self.iteration_repo.update_completion(iteration_id, iteration_record)

            return iteration_record

        except Exception as e:
            # Log error
            self.iteration_repo.update_error(iteration_id, str(e))
            self.llm_repo.update_error(request_id, str(e))
            raise

    def _run_orchestrator_decision(
        self,
        researcher: ResearcherRecord,
        iteration_num: int,
        iterations: list[ResearchIterationRecord],
    ) -> Optional[OrchestratorDecision]:
        """
        Run orchestrator agent to decide if more research is needed.

        Args:
            researcher: Researcher record
            iteration_num: Current iteration number
            iterations: List of all iterations

        Returns:
            Orchestrator decision if successful
        """
        try:
            # Create LLM request record
            request_id = str(uuid.uuid4())
            llm_request = LLMRequestRecord(
                request_id=request_id,
                researcher_id=researcher.researcher_id,
                request_type="orchestrator_decision",
                model_name=self.model_name,
                prompt=self._build_orchestrator_prompt(
                    researcher, iteration_num, iterations
                ),
                status=ProcessingStatus.PENDING,
            )
            self.llm_repo.create(llm_request)

            # Run the task
            start_time = time.time()
            result = self._execute_orchestrator_task(llm_request.prompt)
            latency_ms = int((time.time() - start_time) * 1000)

            # Parse the output
            decision = OrchestratorDecision.model_validate_json(
                result.samples[0].output.completion
            )

            # Update LLM request with response
            self.llm_repo.update_response(
                request_id=request_id,
                response=result.samples[0].output.completion,
                tokens_prompt=0,
                tokens_completion=0,
                latency_ms=latency_ms,
                cost_usd=0.0,
            )

            # Save decision record
            decision_record = OrchestratorDecisionRecord(
                researcher_id=researcher.researcher_id,
                iteration_number=iteration_num,
                continue_research=decision.continue_research,
                rationale=decision.rationale,
                completeness_score=decision.assessment.completeness_score,
                filled_fields=decision.assessment.filled_fields,
                missing_fields=decision.assessment.missing_fields,
                next_focus_areas=decision.next_focus_areas,
            )
            self.orchestrator_repo.create(decision_record)

            return decision

        except Exception as e:
            self.llm_repo.update_error(request_id, str(e))
            raise

    def _build_research_prompt(
        self,
        researcher: ResearcherRecord,
        iteration_num: int,
        previous_iterations: list[ResearchIterationRecord],
    ) -> str:
        """Build prompt for research iteration."""
        # Build summary of previous iterations
        previous_summary = "none"
        if previous_iterations:
            previous_summary = "\n".join(
                [
                    f"Iteration {i.iteration_number}: "
                    f"Searched {i.step_3_location_selection.selected_location if i.step_3_location_selection else 'unknown'}, "
                    f"Found {len(i.step_8_extracted_data.publications) if i.step_8_extracted_data else 0} publications"
                    for i in previous_iterations
                ]
            )

        # Build known info summary
        known_info = f"- Name: {researcher.name}\n"
        if researcher.field:
            known_info += f"- Field: {researcher.field}\n"
        if researcher.known_info:
            for key, value in researcher.known_info.items():
                known_info += f"- {key}: {value}\n"

        prompt = f"""You are a research profiling agent. Research {researcher.name}, a highly cited researcher.

This is iteration {iteration_num}. Previous iterations: {previous_summary}

Current known information about the researcher:
{known_info}

Your task is to perform ONE research iteration following these steps:
1. Reflect on the current researcher data
2. Review results of previous iterations
3. Select a place to search (not overlapping with previous results)
4. Select the most appropriate language for the search
5. Use web search to find the correct URL for the selected place
6. Formulate a detailed search strategy for this website
7. Use the information to simulate browsing (describe what pages you would visit)
8. Extract relevant data about the researcher

Provide your response as a structured JSON output following the ResearchIterationOutput schema."""

        return prompt

    def _build_orchestrator_prompt(
        self,
        researcher: ResearcherRecord,
        iteration_num: int,
        iterations: list[ResearchIterationRecord],
    ) -> str:
        """Build prompt for orchestrator decision."""
        # Summarize iterations
        iteration_summary = "\n\n".join(
            [
                f"Iteration {i.iteration_number}: Collected data from "
                f"{i.step_3_location_selection.selected_location if i.step_3_location_selection else 'unknown'}\n"
                f"- Publications: {len(i.step_8_extracted_data.publications) if i.step_8_extracted_data else 0}\n"
                f"- Affiliations: {', '.join(i.step_8_extracted_data.affiliations[:3]) if i.step_8_extracted_data else 'none'}"
                for i in iterations
            ]
        )

        prompt = f"""You are an orchestrator agent managing a researcher profiling system.

You have completed {iteration_num} iteration(s) of research on {researcher.name}:

{iteration_summary}

Assess the completeness of the researcher profile and decide whether to continue research or if the dataset is sufficiently filled.

Provide your response as a structured JSON output following the OrchestratorDecision schema."""

        return prompt

    def _execute_research_task(self, prompt: str):
        """Execute research task with Inspect AI."""
        model = get_model(self.model_name)

        @task
        def research_task():
            solvers = []
            if self.enable_tools:
                solvers.append(
                    use_tools([web_search(providers="tavily"), *web_browser()])
                )
            solvers.append(generate())

            return Task(
                dataset=[Sample(input=prompt, target="")],
                solver=solvers,
                config=GenerateConfig(
                    response_schema=ResponseSchema(
                        name="ResearchIterationOutput",
                        json_schema=json_schema(ResearchIterationOutput),
                        strict=True,
                    ),
                    max_tokens=8192,
                ),
            )

        return eval(research_task(), model=model)[0]

    def _execute_orchestrator_task(self, prompt: str):
        """Execute orchestrator task with Inspect AI."""
        model = get_model(self.model_name)

        @task
        def orchestrator_task():
            return Task(
                dataset=[Sample(input=prompt, target="")],
                solver=generate(),
                config=GenerateConfig(
                    response_schema=ResponseSchema(
                        name="OrchestratorDecision",
                        json_schema=json_schema(OrchestratorDecision),
                        strict=True,
                    ),
                    max_tokens=4096,
                ),
            )

        return eval(orchestrator_task(), model=model)[0]

    def close(self) -> None:
        """Close database connections."""
        self.db.close()
