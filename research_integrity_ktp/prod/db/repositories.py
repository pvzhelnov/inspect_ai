"""
Repository classes for database operations.

Provides high-level interface for CRUD operations on all entities.
"""

import uuid
from datetime import datetime
from typing import Iterator, Optional

from ..models.schemas import (
    LLMRequestRecord,
    OrchestratorDecisionRecord,
    ProcessingBatchRecord,
    ProcessingStatus,
    ResearcherRecord,
    ResearchIterationRecord,
)
from .database import Database


class ResearcherRepository:
    """Repository for researcher operations."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, researcher: ResearcherRecord) -> int:
        """Create a new researcher record."""
        data = {
            "researcher_id": researcher.researcher_id,
            "name": researcher.name,
            "field": researcher.field,
            "known_info": self.db.to_json(researcher.known_info),
            "status": researcher.status.value,
            "priority": researcher.priority,
        }
        return self.db.insert("researchers", data)

    def get_by_id(self, researcher_id: str) -> Optional[ResearcherRecord]:
        """Get researcher by ID."""
        row = self.db.fetchone(
            "SELECT * FROM researchers WHERE researcher_id = ?", (researcher_id,)
        )
        if not row:
            return None

        row["known_info"] = self.db.from_json(row.get("known_info"))
        row["status"] = ProcessingStatus(row["status"])
        return ResearcherRecord(**row)

    def get_pending(self, limit: int = 100) -> list[ResearcherRecord]:
        """Get pending researchers ordered by priority."""
        rows = self.db.fetchall(
            """
            SELECT * FROM researchers
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        result = []
        for row in rows:
            row["known_info"] = self.db.from_json(row.get("known_info"))
            row["status"] = ProcessingStatus(row["status"])
            result.append(ResearcherRecord(**row))
        return result

    def update_status(
        self, researcher_id: str, status: ProcessingStatus
    ) -> None:
        """Update researcher status."""
        self.db.update(
            "researchers",
            {"status": status.value, "updated_at": datetime.now().isoformat()},
            "researcher_id = :researcher_id",
            {"researcher_id": researcher_id},
        )

    def batch_iterator(
        self, batch_size: int = 1000, status: Optional[ProcessingStatus] = None
    ) -> Iterator[list[ResearcherRecord]]:
        """Iterate over researchers in batches."""
        sql = "SELECT * FROM researchers"
        params = None

        if status:
            sql += " WHERE status = ?"
            params = (status.value,)

        sql += " ORDER BY id"

        for batch in self.db.batch_iterator(sql, batch_size, params):
            result = []
            for row in batch:
                row["known_info"] = self.db.from_json(row.get("known_info"))
                row["status"] = ProcessingStatus(row["status"])
                result.append(ResearcherRecord(**row))
            yield result

    def count(self, status: Optional[ProcessingStatus] = None) -> int:
        """Count researchers by status."""
        if status:
            return self.db.count(
                "researchers", "status = ?", (status.value,)
            )
        return self.db.count("researchers")


class ResearchIterationRepository:
    """Repository for research iteration operations."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, iteration: ResearchIterationRecord) -> int:
        """Create a new research iteration record."""
        data = {
            "researcher_id": iteration.researcher_id,
            "iteration_number": iteration.iteration_number,
            "step_1_reflection": self.db.to_json(
                iteration.step_1_reflection.model_dump()
                if iteration.step_1_reflection
                else None
            ),
            "step_2_previous_review": self.db.to_json(
                iteration.step_2_previous_review.model_dump()
                if iteration.step_2_previous_review
                else None
            ),
            "step_3_location_selection": self.db.to_json(
                iteration.step_3_location_selection.model_dump()
                if iteration.step_3_location_selection
                else None
            ),
            "step_4_language_selection": self.db.to_json(
                iteration.step_4_language_selection.model_dump()
                if iteration.step_4_language_selection
                else None
            ),
            "step_5_web_search": self.db.to_json(
                iteration.step_5_web_search.model_dump()
                if iteration.step_5_web_search
                else None
            ),
            "step_6_search_strategy": self.db.to_json(
                iteration.step_6_search_strategy.model_dump()
                if iteration.step_6_search_strategy
                else None
            ),
            "step_7_browsing": self.db.to_json(
                iteration.step_7_browsing.model_dump()
                if iteration.step_7_browsing
                else None
            ),
            "step_8_extracted_data": self.db.to_json(
                iteration.step_8_extracted_data.model_dump()
                if iteration.step_8_extracted_data
                else None
            ),
            "status": iteration.status.value,
            "error_message": iteration.error_message,
            "started_at": (
                iteration.started_at.isoformat() if iteration.started_at else None
            ),
        }
        return self.db.insert("research_iterations", data)

    def get_by_researcher(
        self, researcher_id: str
    ) -> list[ResearchIterationRecord]:
        """Get all iterations for a researcher."""
        rows = self.db.fetchall(
            """
            SELECT * FROM research_iterations
            WHERE researcher_id = ?
            ORDER BY iteration_number
            """,
            (researcher_id,),
        )
        return [self._row_to_record(row) for row in rows]

    def get_latest(
        self, researcher_id: str
    ) -> Optional[ResearchIterationRecord]:
        """Get the latest iteration for a researcher."""
        row = self.db.fetchone(
            """
            SELECT * FROM research_iterations
            WHERE researcher_id = ?
            ORDER BY iteration_number DESC
            LIMIT 1
            """,
            (researcher_id,),
        )
        return self._row_to_record(row) if row else None

    def update_completion(
        self, iteration_id: int, iteration: ResearchIterationRecord
    ) -> None:
        """Update iteration with completion data."""
        data = {
            "step_1_reflection": self.db.to_json(
                iteration.step_1_reflection.model_dump()
            ),
            "step_2_previous_review": self.db.to_json(
                iteration.step_2_previous_review.model_dump()
            ),
            "step_3_location_selection": self.db.to_json(
                iteration.step_3_location_selection.model_dump()
            ),
            "step_4_language_selection": self.db.to_json(
                iteration.step_4_language_selection.model_dump()
            ),
            "step_5_web_search": self.db.to_json(
                iteration.step_5_web_search.model_dump()
            ),
            "step_6_search_strategy": self.db.to_json(
                iteration.step_6_search_strategy.model_dump()
            ),
            "step_7_browsing": self.db.to_json(
                iteration.step_7_browsing.model_dump()
            ),
            "step_8_extracted_data": self.db.to_json(
                iteration.step_8_extracted_data.model_dump()
            ),
            "status": ProcessingStatus.COMPLETED.value,
            "completed_at": datetime.now().isoformat(),
        }
        self.db.update(
            "research_iterations", data, "id = :id", {"id": iteration_id}
        )

    def update_error(self, iteration_id: int, error_message: str) -> None:
        """Update iteration with error."""
        data = {
            "status": ProcessingStatus.FAILED.value,
            "error_message": error_message,
            "completed_at": datetime.now().isoformat(),
        }
        self.db.update(
            "research_iterations", data, "id = :id", {"id": iteration_id}
        )

    def _row_to_record(self, row: dict) -> ResearchIterationRecord:
        """Convert database row to record."""
        from ..models.schemas import (
            BrowsingResults,
            ExtractedData,
            LanguageSelection,
            PreviousIterationReview,
            ResearcherReflection,
            SearchLocationSelection,
            SearchStrategy,
            WebSearchResults,
        )

        return ResearchIterationRecord(
            id=row["id"],
            researcher_id=row["researcher_id"],
            iteration_number=row["iteration_number"],
            step_1_reflection=(
                ResearcherReflection(**self.db.from_json(row["step_1_reflection"]))
                if row.get("step_1_reflection")
                else None
            ),
            step_2_previous_review=(
                PreviousIterationReview(
                    **self.db.from_json(row["step_2_previous_review"])
                )
                if row.get("step_2_previous_review")
                else None
            ),
            step_3_location_selection=(
                SearchLocationSelection(
                    **self.db.from_json(row["step_3_location_selection"])
                )
                if row.get("step_3_location_selection")
                else None
            ),
            step_4_language_selection=(
                LanguageSelection(
                    **self.db.from_json(row["step_4_language_selection"])
                )
                if row.get("step_4_language_selection")
                else None
            ),
            step_5_web_search=(
                WebSearchResults(**self.db.from_json(row["step_5_web_search"]))
                if row.get("step_5_web_search")
                else None
            ),
            step_6_search_strategy=(
                SearchStrategy(**self.db.from_json(row["step_6_search_strategy"]))
                if row.get("step_6_search_strategy")
                else None
            ),
            step_7_browsing=(
                BrowsingResults(**self.db.from_json(row["step_7_browsing"]))
                if row.get("step_7_browsing")
                else None
            ),
            step_8_extracted_data=(
                ExtractedData(**self.db.from_json(row["step_8_extracted_data"]))
                if row.get("step_8_extracted_data")
                else None
            ),
            status=ProcessingStatus(row["status"]),
            error_message=row.get("error_message"),
            started_at=(
                datetime.fromisoformat(row["started_at"]) if row.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row.get("completed_at")
                else None
            ),
            created_at=(
                datetime.fromisoformat(row["created_at"])
                if row.get("created_at")
                else None
            ),
        )


class LLMRequestRepository:
    """Repository for LLM request operations."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, request: LLMRequestRecord) -> int:
        """Create a new LLM request record."""
        data = {
            "request_id": request.request_id or str(uuid.uuid4()),
            "researcher_id": request.researcher_id,
            "iteration_id": request.iteration_id,
            "request_type": request.request_type,
            "model_name": request.model_name,
            "prompt": request.prompt,
            "status": request.status.value,
        }
        return self.db.insert("llm_requests", data)

    def update_response(
        self,
        request_id: str,
        response: str,
        tokens_prompt: int,
        tokens_completion: int,
        latency_ms: int,
        cost_usd: float = 0.0,
    ) -> None:
        """Update request with response data."""
        data = {
            "response": response,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_prompt + tokens_completion,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "status": ProcessingStatus.COMPLETED.value,
            "completed_at": datetime.now().isoformat(),
        }
        self.db.update(
            "llm_requests",
            data,
            "request_id = :request_id",
            {"request_id": request_id},
        )

    def update_error(self, request_id: str, error_message: str) -> None:
        """Update request with error."""
        data = {
            "status": ProcessingStatus.FAILED.value,
            "error_message": error_message,
            "completed_at": datetime.now().isoformat(),
        }
        self.db.update(
            "llm_requests",
            data,
            "request_id = :request_id",
            {"request_id": request_id},
        )

    def get_stats(self) -> dict:
        """Get request statistics."""
        stats = self.db.fetchone(
            """
            SELECT
                COUNT(*) as total_requests,
                SUM(tokens_total) as total_tokens,
                SUM(cost_usd) as total_cost,
                AVG(latency_ms) as avg_latency
            FROM llm_requests
            WHERE status = 'completed'
            """
        )
        return stats or {}


class OrchestratorDecisionRepository:
    """Repository for orchestrator decision operations."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, decision: OrchestratorDecisionRecord) -> int:
        """Create a new orchestrator decision record."""
        data = {
            "researcher_id": decision.researcher_id,
            "iteration_number": decision.iteration_number,
            "continue_research": int(decision.continue_research),
            "rationale": decision.rationale,
            "completeness_score": decision.completeness_score,
            "filled_fields": self.db.to_json(decision.filled_fields),
            "missing_fields": self.db.to_json(decision.missing_fields),
            "next_focus_areas": self.db.to_json(decision.next_focus_areas),
        }
        return self.db.insert("orchestrator_decisions", data)

    def get_latest(
        self, researcher_id: str
    ) -> Optional[OrchestratorDecisionRecord]:
        """Get the latest decision for a researcher."""
        row = self.db.fetchone(
            """
            SELECT * FROM orchestrator_decisions
            WHERE researcher_id = ?
            ORDER BY iteration_number DESC
            LIMIT 1
            """,
            (researcher_id,),
        )
        if not row:
            return None

        return OrchestratorDecisionRecord(
            id=row["id"],
            researcher_id=row["researcher_id"],
            iteration_number=row["iteration_number"],
            continue_research=bool(row["continue_research"]),
            rationale=row["rationale"],
            completeness_score=row["completeness_score"],
            filled_fields=self.db.from_json(row["filled_fields"]),
            missing_fields=self.db.from_json(row["missing_fields"]),
            next_focus_areas=self.db.from_json(row["next_focus_areas"]),
            created_at=(
                datetime.fromisoformat(row["created_at"])
                if row.get("created_at")
                else None
            ),
        )


class ProcessingBatchRepository:
    """Repository for processing batch operations."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, batch: ProcessingBatchRecord) -> int:
        """Create a new batch record."""
        data = {
            "batch_name": batch.batch_name,
            "total_samples": batch.total_samples,
            "worker_count": batch.worker_count,
            "status": batch.status.value,
        }
        return self.db.insert("processing_batches", data)

    def update_progress(
        self,
        batch_name: str,
        processed: int,
        successful: int,
        failed: int,
    ) -> None:
        """Update batch progress."""
        data = {
            "processed_samples": processed,
            "successful_samples": successful,
            "failed_samples": failed,
            "updated_at": datetime.now().isoformat(),
        }
        self.db.update(
            "processing_batches",
            data,
            "batch_name = :batch_name",
            {"batch_name": batch_name},
        )

    def update_status(
        self, batch_name: str, status: ProcessingStatus
    ) -> None:
        """Update batch status."""
        data = {
            "status": status.value,
            "updated_at": datetime.now().isoformat(),
        }

        if status == ProcessingStatus.PROCESSING:
            data["started_at"] = datetime.now().isoformat()
        elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            data["completed_at"] = datetime.now().isoformat()

        self.db.update(
            "processing_batches",
            data,
            "batch_name = :batch_name",
            {"batch_name": batch_name},
        )

    def get_by_name(self, batch_name: str) -> Optional[ProcessingBatchRecord]:
        """Get batch by name."""
        row = self.db.fetchone(
            "SELECT * FROM processing_batches WHERE batch_name = ?",
            (batch_name,),
        )
        if not row:
            return None

        row["status"] = ProcessingStatus(row["status"])
        return ProcessingBatchRecord(**row)
