#!/usr/bin/env python3
"""
Query the researcher profiling database.

Usage:
    python query_db.py --stats
    python query_db.py --researcher hinton_001
    python query_db.py --export results.json
"""

import argparse
import json
import sys
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


def print_stats(db: Database):
    """Print database statistics."""
    print("\n" + "=" * 70)
    print("DATABASE STATISTICS")
    print("=" * 70)

    # Researcher stats
    researcher_repo = ResearcherRepository(db)
    total = researcher_repo.count()
    pending = researcher_repo.count(status="pending")
    processing = researcher_repo.count(status="processing")
    completed = researcher_repo.count(status="completed")
    failed = researcher_repo.count(status="failed")

    print(f"\nResearchers:")
    print(f"  Total:      {total}")
    print(f"  Pending:    {pending}")
    print(f"  Processing: {processing}")
    print(f"  Completed:  {completed}")
    print(f"  Failed:     {failed}")

    # Iteration stats
    stats = db.fetchone(
        """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
        FROM research_iterations
        """
    )
    print(f"\nResearch Iterations:")
    print(f"  Total:     {stats['total']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed:    {stats['failed']}")

    # LLM request stats
    llm_repo = LLMRequestRepository(db)
    llm_stats = llm_repo.get_stats()
    print(f"\nLLM Requests:")
    print(f"  Total Requests:   {llm_stats.get('total_requests', 0)}")
    print(f"  Total Tokens:     {llm_stats.get('total_tokens', 0):,}")
    print(f"  Total Cost (USD): ${llm_stats.get('total_cost', 0):.2f}")
    print(f"  Avg Latency (ms): {llm_stats.get('avg_latency', 0):.0f}")

    # Batch stats
    batches = db.fetchall(
        """
        SELECT * FROM processing_batches
        ORDER BY created_at DESC
        LIMIT 5
        """
    )
    if batches:
        print(f"\nRecent Batches:")
        for batch in batches:
            progress = (
                batch["processed_samples"] / batch["total_samples"] * 100
                if batch["total_samples"] > 0
                else 0
            )
            print(f"  {batch['batch_name']}: {progress:.1f}% ({batch['status']})")

    print("=" * 70)


def print_researcher(db: Database, researcher_id: str):
    """Print detailed researcher information."""
    researcher_repo = ResearcherRepository(db)
    iteration_repo = ResearchIterationRepository(db)
    orchestrator_repo = OrchestratorDecisionRepository(db)

    # Get researcher
    researcher = researcher_repo.get_by_id(researcher_id)
    if not researcher:
        print(f"Researcher not found: {researcher_id}")
        return

    print("\n" + "=" * 70)
    print(f"RESEARCHER: {researcher.name}")
    print("=" * 70)
    print(f"ID:     {researcher.researcher_id}")
    print(f"Field:  {researcher.field}")
    print(f"Status: {researcher.status.value}")

    if researcher.known_info:
        print(f"\nKnown Information:")
        for key, value in researcher.known_info.items():
            print(f"  {key}: {value}")

    # Get iterations
    iterations = iteration_repo.get_by_researcher(researcher_id)
    print(f"\nIterations: {len(iterations)}")

    for iteration in iterations:
        print(f"\n  Iteration {iteration.iteration_number} ({iteration.status.value}):")

        if iteration.step_3_location_selection:
            print(f"    Location: {iteration.step_3_location_selection.selected_location}")

        if iteration.step_5_web_search:
            print(f"    Search: {iteration.step_5_web_search.search_query}")
            print(f"    URL: {iteration.step_5_web_search.target_url}")

        if iteration.step_8_extracted_data:
            data = iteration.step_8_extracted_data
            print(f"    Publications: {len(data.publications)}")
            print(f"    Affiliations: {', '.join(data.affiliations[:3])}")
            if data.h_index:
                print(f"    H-index: {data.h_index}")
            if data.total_citations:
                print(f"    Citations: {data.total_citations}")

    # Get orchestrator decisions
    decision = orchestrator_repo.get_latest(researcher_id)
    if decision:
        print(f"\n  Latest Decision:")
        print(f"    Continue: {decision.continue_research}")
        print(f"    Completeness: {decision.completeness_score:.2%}")
        print(f"    Rationale: {decision.rationale[:100]}...")

    print("=" * 70)


def export_results(db: Database, output_file: Path):
    """Export results to JSON file."""
    researcher_repo = ResearcherRepository(db)
    iteration_repo = ResearchIterationRepository(db)

    print(f"Exporting results to {output_file}...")

    results = []

    # Get all completed researchers
    for batch in researcher_repo.batch_iterator(status="completed"):
        for researcher in batch:
            # Get iterations
            iterations = iteration_repo.get_by_researcher(researcher.researcher_id)

            researcher_data = {
                "researcher_id": researcher.researcher_id,
                "name": researcher.name,
                "field": researcher.field,
                "status": researcher.status.value,
                "iterations": [],
            }

            for iteration in iterations:
                iteration_data = {
                    "iteration_number": iteration.iteration_number,
                    "status": iteration.status.value,
                }

                if iteration.step_8_extracted_data:
                    data = iteration.step_8_extracted_data
                    iteration_data["extracted_data"] = {
                        "publications_count": len(data.publications),
                        "publications": [p.model_dump() for p in data.publications],
                        "affiliations": data.affiliations,
                        "research_areas": data.research_areas,
                        "h_index": data.h_index,
                        "total_citations": data.total_citations,
                        "awards": data.awards,
                        "collaborators": data.collaborators,
                    }

                researcher_data["iterations"].append(iteration_data)

            results.append(researcher_data)

    # Write to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Exported {len(results)} researchers to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query researcher profiling database"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/researcher_profiles.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics",
    )
    parser.add_argument(
        "--researcher",
        type=str,
        help="Show details for specific researcher ID",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export results to JSON file",
    )

    args = parser.parse_args()

    # Check database exists
    if not args.db_path.exists():
        print(f"Database not found: {args.db_path}")
        print("Run batch processing first to create the database.")
        return 1

    # Connect to database
    db_config = DatabaseConfig(db_path=args.db_path)
    db = Database(db_config)

    try:
        if args.stats:
            print_stats(db)

        if args.researcher:
            print_researcher(db, args.researcher)

        if args.export:
            export_results(db, args.export)

        if not (args.stats or args.researcher or args.export):
            parser.print_help()

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
