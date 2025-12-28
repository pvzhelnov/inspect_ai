#!/usr/bin/env python3
"""
Main script for running batch processing of researcher profiles.

Usage:
    python run_batch.py --input samples.jsonl --batch-name batch1 --workers 4

Features:
- Load millions of samples from file
- Parallel processing with multiple workers
- Progress tracking in SQLite database
- Resume capability after interruption
- Comprehensive logging
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from research_integrity_ktp.prod.agents.researcher_profiling import (
    ResearcherProfilingAgent,
)
from research_integrity_ktp.prod.models.schemas import ResearcherRecord
from research_integrity_ktp.prod.utils.batch_processor import BatchProcessor

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("batch_processing.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def parse_jsonl_line(line: str) -> ResearcherRecord:
    """
    Parse a JSONL line into ResearcherRecord.

    Expected format:
    {"researcher_id": "...", "name": "...", "field": "...", "known_info": {...}}
    """
    data = json.loads(line)
    return ResearcherRecord(
        researcher_id=data["researcher_id"],
        name=data["name"],
        field=data.get("field"),
        known_info=data.get("known_info", {}),
        priority=data.get("priority", 0),
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process researcher profiles with Inspect AI"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file with researcher samples (JSONL format)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/researcher_profiles.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--batch-name",
        type=str,
        required=True,
        help="Unique name for this batch job",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/qwen/qwen3-coder:free",
        help="LLM model to use",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum research iterations per researcher",
    )
    parser.add_argument(
        "--enable-tools",
        action="store_true",
        default=False,
        help="Enable web_search and web_browser tools",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only load samples into database, don't process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing from where it left off",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("RESEARCHER PROFILING BATCH PROCESSOR")
    logger.info("=" * 70)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Batch name: {args.batch_name}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Tools enabled: {args.enable_tools}")
    logger.info("=" * 70)

    # Initialize batch processor
    processor = BatchProcessor(
        db_path=args.db_path,
        batch_name=args.batch_name,
        worker_count=args.workers,
        batch_size=args.batch_size,
    )

    try:
        # Load samples if not resuming
        if not args.resume:
            if not args.input.exists():
                logger.error(f"Input file not found: {args.input}")
                return 1

            processor.load_samples_from_file(args.input, parse_jsonl_line)
            processor.create_batch()

        if args.load_only:
            logger.info("Samples loaded successfully. Exiting (--load-only flag).")
            return 0

        # Initialize researcher profiling agent
        agent = ResearcherProfilingAgent(
            db_path=args.db_path,
            model_name=args.model,
            max_iterations=args.max_iterations,
            enable_tools=args.enable_tools,
        )

        # Define process function
        def process_researcher(researcher: ResearcherRecord) -> None:
            """Process a single researcher."""
            logger.info(f"Processing researcher: {researcher.name}")
            agent.process_researcher(researcher)

        # Start batch processing
        import asyncio

        asyncio.run(processor.process_batch(process_researcher))

        # Print final stats
        stats = processor.get_stats()
        logger.info("\n" + "=" * 70)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 70)
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 70)

        # Get LLM request stats
        llm_stats = agent.llm_repo.get_stats()
        logger.info("\nLLM REQUEST STATISTICS")
        logger.info("=" * 70)
        for key, value in llm_stats.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 70)

        # Cleanup
        agent.close()
        processor.close()

        return 0

    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user. Progress saved to database.")
        logger.info("Use --resume flag to continue from where you left off.")
        return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
