"""
Pydantic models for researcher profiling production system.

Includes all schemas for:
- Research iteration steps (8-step process)
- Orchestrator decisions
- Database records
- Batch processing
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ==============================================================================
# Research Agent Schemas (8-step process)
# ==============================================================================


class ResearcherReflection(BaseModel):
    """Step 1: Reflection on the researcher's current data state."""

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
    """Step 2: Review of previous research iterations."""

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
    """Step 3: Selection of where to search next."""

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
    """Step 4: Selection of search language."""

    language: str = Field(
        ..., description="Language code (e.g., 'en', 'de', 'fr', 'zh', 'ja')"
    )
    language_name: str = Field(..., description="Full language name (e.g., 'English')")
    rationale: str = Field(
        ..., description="Why this language is most appropriate for this search"
    )


class WebSearchResults(BaseModel):
    """Step 5: Web search results for finding the target URL."""

    search_query: str = Field(..., description="The search query used")
    target_url: str = Field(
        ..., description="The URL identified as most relevant for research"
    )
    alternative_urls: list[str] = Field(
        default_factory=list, description="Alternative URLs that might be useful"
    )
    search_results_count: int = Field(
        ..., description="Total number of search results found"
    )


class SearchStrategy(BaseModel):
    """Step 6: Detailed search strategy for browsing."""

    navigation_plan: list[str] = Field(
        ..., description="Step-by-step plan for navigating the selected website"
    )
    data_extraction_focus: list[str] = Field(
        ..., description="Specific data points to extract from each page"
    )
    estimated_pages_to_visit: int = Field(
        ..., description="Estimated number of pages to visit"
    )


class PageVisit(BaseModel):
    """Information about a visited page."""

    url: str = Field(..., description="URL of the visited page")
    page_title: str = Field(default="", description="Title of the page")
    content_summary: str = Field(..., description="Summary of page content")
    data_found: list[str] = Field(
        default_factory=list, description="Data points found on this page"
    )


class BrowsingResults(BaseModel):
    """Step 7: Results from browsing web pages."""

    pages_visited: list[PageVisit] = Field(
        ..., description="List of pages visited during research"
    )
    total_pages_visited: int = Field(..., description="Total number of pages visited")
    browsing_notes: str = Field(..., description="Overall notes from browsing session")


class Publication(BaseModel):
    """A single publication."""

    title: str = Field(..., description="Publication title")
    year: Optional[int] = Field(None, description="Publication year")
    citations: Optional[int] = Field(None, description="Number of citations")
    venue: Optional[str] = Field(None, description="Publication venue")


class ExtractedData(BaseModel):
    """Step 8: Data extracted from research."""

    publications: list[Publication] = Field(
        default_factory=list, description="List of publications found"
    )
    affiliations: list[str] = Field(
        default_factory=list, description="Current and past affiliations"
    )
    research_areas: list[str] = Field(
        default_factory=list, description="Research areas/topics"
    )
    h_index: Optional[int] = Field(None, description="H-index if available")
    total_citations: Optional[int] = Field(
        None, description="Total citation count if available"
    )
    awards: list[str] = Field(default_factory=list, description="Awards and honors")
    collaborators: list[str] = Field(
        default_factory=list, description="Key collaborators"
    )
    other_data: dict[str, Any] = Field(
        default_factory=dict, description="Any other relevant data"
    )


class ResearchIterationOutput(BaseModel):
    """Complete output from one research iteration (all 8 steps)."""

    step_1_reflection: ResearcherReflection
    step_2_previous_review: PreviousIterationReview
    step_3_location_selection: SearchLocationSelection
    step_4_language_selection: LanguageSelection
    step_5_web_search: WebSearchResults
    step_6_search_strategy: SearchStrategy
    step_7_browsing: BrowsingResults
    step_8_extracted_data: ExtractedData


# ==============================================================================
# Orchestrator Schemas
# ==============================================================================


class ProfileCompletenessAssessment(BaseModel):
    """Assessment of researcher profile completeness."""

    completeness_score: float = Field(
        ..., description="Completeness score from 0.0 to 1.0"
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


# ==============================================================================
# Database Record Models
# ==============================================================================


class ProcessingStatus(str, Enum):
    """Processing status enum."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearcherRecord(BaseModel):
    """Database record for a researcher."""

    id: Optional[int] = None
    researcher_id: str
    name: str
    field: Optional[str] = None
    known_info: Optional[dict[str, Any]] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    priority: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ResearchIterationRecord(BaseModel):
    """Database record for a research iteration."""

    id: Optional[int] = None
    researcher_id: str
    iteration_number: int
    step_1_reflection: Optional[ResearcherReflection] = None
    step_2_previous_review: Optional[PreviousIterationReview] = None
    step_3_location_selection: Optional[SearchLocationSelection] = None
    step_4_language_selection: Optional[LanguageSelection] = None
    step_5_web_search: Optional[WebSearchResults] = None
    step_6_search_strategy: Optional[SearchStrategy] = None
    step_7_browsing: Optional[BrowsingResults] = None
    step_8_extracted_data: Optional[ExtractedData] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class LLMRequestRecord(BaseModel):
    """Database record for an LLM request."""

    id: Optional[int] = None
    request_id: str
    researcher_id: Optional[str] = None
    iteration_id: Optional[int] = None
    request_type: str  # research_iteration, orchestrator_decision
    model_name: str
    prompt: str
    response: Optional[str] = None
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    tokens_total: Optional[int] = None
    cost_usd: float = 0.0
    latency_ms: Optional[int] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class OrchestratorDecisionRecord(BaseModel):
    """Database record for an orchestrator decision."""

    id: Optional[int] = None
    researcher_id: str
    iteration_number: int
    continue_research: bool
    rationale: str
    completeness_score: float
    filled_fields: list[str]
    missing_fields: list[str]
    next_focus_areas: list[str]
    created_at: Optional[datetime] = None


class ProcessingBatchRecord(BaseModel):
    """Database record for a processing batch."""

    id: Optional[int] = None
    batch_name: str
    total_samples: int
    processed_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    worker_count: int = 1
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
