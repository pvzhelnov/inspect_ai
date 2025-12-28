#!/usr/bin/env python3
"""
Test the production system without making API calls.

Verifies:
- Database initialization
- Schema creation
- Sample loading
- Repository operations
"""

import sys
import tempfile
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from research_integrity_ktp.prod.db.database import Database, DatabaseConfig
from research_integrity_ktp.prod.db.repositories import (
    LLMRequestRepository,
    OrchestratorDecisionRepository,
    ProcessingBatchRepository,
    ResearcherRepository,
    ResearchIterationRepository,
)
from research_integrity_ktp.prod.models.schemas import (
    LLMRequestRecord,
    OrchestratorDecisionRecord,
    ProcessingBatchRecord,
    ProcessingStatus,
    ProfileCompletenessAssessment,
    ResearcherRecord,
    ResearchIterationRecord,
)


def test_database_init():
    """Test database initialization."""
    print("\n" + "=" * 70)
    print("TEST 1: Database Initialization")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Initialize database
        db_config = DatabaseConfig(db_path=db_path)
        db = Database(db_config)

        # Check tables exist
        tables = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = [t["name"] for t in tables]

        print(f"✓ Database created at {db_path}")
        print(f"✓ Found {len(table_names)} tables:")
        for name in table_names:
            print(f"    - {name}")

        expected_tables = [
            "browser_cache",
            "llm_requests",
            "orchestrator_decisions",
            "processing_batches",
            "researchers",
            "research_iterations",
            "system_metrics",
            "web_search_cache",
        ]

        for table in expected_tables:
            assert table in table_names, f"Missing table: {table}"

        print("✓ All expected tables exist")

        db.close()


def test_researcher_repository():
    """Test researcher repository operations."""
    print("\n" + "=" * 70)
    print("TEST 2: Researcher Repository")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_config = DatabaseConfig(db_path=db_path)
        db = Database(db_config)
        repo = ResearcherRepository(db)

        # Create researcher
        researcher = ResearcherRecord(
            researcher_id="test_001",
            name="Test Researcher",
            field="Computer Science",
            known_info={"affiliation": "Test University"},
            priority=5,
        )

        researcher_id = repo.create(researcher)
        print(f"✓ Created researcher with ID: {researcher_id}")

        # Get researcher
        retrieved = repo.get_by_id("test_001")
        assert retrieved is not None
        assert retrieved.name == "Test Researcher"
        print(f"✓ Retrieved researcher: {retrieved.name}")

        # Update status
        repo.update_status("test_001", ProcessingStatus.COMPLETED)
        updated = repo.get_by_id("test_001")
        assert updated.status == ProcessingStatus.COMPLETED
        print(f"✓ Updated status to: {updated.status.value}")

        # Count
        count = repo.count(ProcessingStatus.COMPLETED)
        assert count == 1
        print(f"✓ Count completed researchers: {count}")

        db.close()


def test_iteration_repository():
    """Test research iteration repository operations."""
    print("\n" + "=" * 70)
    print("TEST 3: Research Iteration Repository")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_config = DatabaseConfig(db_path=db_path)
        db = Database(db_config)

        # Create researcher first
        researcher_repo = ResearcherRepository(db)
        researcher = ResearcherRecord(
            researcher_id="test_002",
            name="Another Researcher",
            field="AI",
        )
        researcher_repo.create(researcher)

        # Create iteration
        iteration_repo = ResearchIterationRepository(db)
        iteration = ResearchIterationRecord(
            researcher_id="test_002",
            iteration_number=1,
            status=ProcessingStatus.PENDING,
        )

        iteration_id = iteration_repo.create(iteration)
        print(f"✓ Created iteration with ID: {iteration_id}")

        # Get iterations
        iterations = iteration_repo.get_by_researcher("test_002")
        assert len(iterations) == 1
        print(f"✓ Retrieved {len(iterations)} iteration(s)")

        # Get latest
        latest = iteration_repo.get_latest("test_002")
        assert latest is not None
        assert latest.iteration_number == 1
        print(f"✓ Latest iteration number: {latest.iteration_number}")

        db.close()


def test_batch_repository():
    """Test batch repository operations."""
    print("\n" + "=" * 70)
    print("TEST 4: Batch Repository")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_config = DatabaseConfig(db_path=db_path)
        db = Database(db_config)
        repo = ProcessingBatchRepository(db)

        # Create batch
        batch = ProcessingBatchRecord(
            batch_name="test_batch",
            total_samples=100,
            worker_count=4,
        )

        batch_id = repo.create(batch)
        print(f"✓ Created batch with ID: {batch_id}")

        # Update progress
        repo.update_progress("test_batch", 50, 48, 2)
        updated = repo.get_by_name("test_batch")
        assert updated.processed_samples == 50
        assert updated.successful_samples == 48
        assert updated.failed_samples == 2
        print(f"✓ Updated progress: {updated.processed_samples}/{updated.total_samples}")

        # Update status
        repo.update_status("test_batch", ProcessingStatus.COMPLETED)
        completed = repo.get_by_name("test_batch")
        assert completed.status == ProcessingStatus.COMPLETED
        print(f"✓ Updated status to: {completed.status.value}")

        db.close()


def test_llm_repository():
    """Test LLM request repository operations."""
    print("\n" + "=" * 70)
    print("TEST 5: LLM Request Repository")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_config = DatabaseConfig(db_path=db_path)
        db = Database(db_config)
        repo = LLMRequestRepository(db)

        # Create request
        request = LLMRequestRecord(
            request_id="req_001",
            request_type="test",
            model_name="test-model",
            prompt="Test prompt",
        )

        request_id = repo.create(request)
        print(f"✓ Created LLM request with ID: {request_id}")

        # Update with response
        repo.update_response(
            request_id="req_001",
            response="Test response",
            tokens_prompt=10,
            tokens_completion=20,
            latency_ms=100,
            cost_usd=0.01,
        )
        print("✓ Updated request with response")

        # Get stats
        stats = repo.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 30
        assert stats["total_cost"] == 0.01
        print(f"✓ Stats: {stats['total_requests']} requests, "
              f"{stats['total_tokens']} tokens, ${stats['total_cost']:.2f}")

        db.close()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PRODUCTION SYSTEM TESTS")
    print("=" * 70)

    try:
        test_database_init()
        test_researcher_repository()
        test_iteration_repository()
        test_batch_repository()
        test_llm_repository()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
