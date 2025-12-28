"""
Batch processor for handling millions of researcher samples.

Provides:
- Parallel processing with configurable workers
- Progress tracking and resume capability
- Error handling and retry logic
- Rate limiting for API calls
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from ..db.database import Database, DatabaseConfig
from ..db.repositories import (
    ProcessingBatchRepository,
    ResearcherRepository,
)
from ..models.schemas import (
    ProcessingBatchRecord,
    ProcessingStatus,
    ResearcherRecord,
)

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for handling millions of researcher samples.

    Features:
    - Parallel processing with multiple workers
    - Progress tracking in database
    - Resume capability after interruption
    - Rate limiting for API calls
    - Error handling with configurable retries
    """

    def __init__(
        self,
        db_path: Path,
        batch_name: str,
        worker_count: int = 4,
        batch_size: int = 100,
        max_retries: int = 3,
        rate_limit_per_minute: int = 60,
    ):
        """
        Initialize batch processor.

        Args:
            db_path: Path to SQLite database
            batch_name: Unique name for this batch job
            worker_count: Number of parallel workers
            batch_size: Number of samples per batch
            max_retries: Maximum retry attempts for failed samples
            rate_limit_per_minute: Maximum API calls per minute
        """
        self.db_path = db_path
        self.batch_name = batch_name
        self.worker_count = worker_count
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.rate_limit_per_minute = rate_limit_per_minute

        # Initialize database
        db_config = DatabaseConfig(db_path=db_path, pool_size=worker_count * 2)
        self.db = Database(db_config)

        # Initialize repositories
        self.researcher_repo = ResearcherRepository(self.db)
        self.batch_repo = ProcessingBatchRepository(self.db)

        # Progress tracking
        self.processed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        self.start_time: Optional[float] = None

        # Rate limiting
        self._api_call_times: list[float] = []
        self._rate_limit_lock = asyncio.Lock()

    def load_samples(
        self, samples: list[ResearcherRecord]
    ) -> None:
        """
        Load researcher samples into database.

        Args:
            samples: List of researcher records to process
        """
        logger.info(f"Loading {len(samples)} samples into database...")

        # Batch insert for better performance
        batch_data = []
        for sample in samples:
            batch_data.append(
                {
                    "researcher_id": sample.researcher_id,
                    "name": sample.name,
                    "field": sample.field,
                    "known_info": self.db.to_json(sample.known_info),
                    "status": ProcessingStatus.PENDING.value,
                    "priority": sample.priority,
                }
            )

        # Insert in batches
        for i in range(0, len(batch_data), 1000):
            batch = batch_data[i : i + 1000]
            self.db.insert_many("researchers", batch)

        logger.info(f"Loaded {len(samples)} samples successfully")

    def load_samples_from_file(
        self, file_path: Path, parse_fn: Callable[[str], ResearcherRecord]
    ) -> None:
        """
        Load samples from file line by line (for very large files).

        Args:
            file_path: Path to input file
            parse_fn: Function to parse each line into ResearcherRecord
        """
        logger.info(f"Loading samples from {file_path}...")

        batch_data = []
        count = 0

        with open(file_path) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    sample = parse_fn(line)
                    batch_data.append(
                        {
                            "researcher_id": sample.researcher_id,
                            "name": sample.name,
                            "field": sample.field,
                            "known_info": self.db.to_json(sample.known_info),
                            "status": ProcessingStatus.PENDING.value,
                            "priority": sample.priority,
                        }
                    )
                    count += 1

                    # Insert in batches
                    if len(batch_data) >= 1000:
                        self.db.insert_many("researchers", batch_data)
                        batch_data = []
                        logger.info(f"Loaded {count} samples...")

                except Exception as e:
                    logger.error(f"Error parsing line: {e}")

            # Insert remaining
            if batch_data:
                self.db.insert_many("researchers", batch_data)

        logger.info(f"Loaded {count} samples from file")

    def create_batch(self) -> int:
        """
        Create a new batch record.

        Returns:
            Batch ID
        """
        total_samples = self.researcher_repo.count(ProcessingStatus.PENDING)

        batch = ProcessingBatchRecord(
            batch_name=self.batch_name,
            total_samples=total_samples,
            worker_count=self.worker_count,
            status=ProcessingStatus.PENDING,
        )

        batch_id = self.batch_repo.create(batch)
        logger.info(
            f"Created batch '{self.batch_name}' with {total_samples} samples"
        )
        return batch_id

    async def process_batch(
        self,
        process_fn: Callable[[ResearcherRecord], None],
        use_multiprocessing: bool = False,
    ) -> None:
        """
        Process all pending samples in parallel.

        Args:
            process_fn: Function to process each researcher sample
            use_multiprocessing: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        """
        # Create or get batch record
        batch = self.batch_repo.get_by_name(self.batch_name)
        if not batch:
            self.create_batch()

        # Update batch status to running
        self.batch_repo.update_status(self.batch_name, ProcessingStatus.PROCESSING)

        self.start_time = time.time()
        logger.info(
            f"Starting batch processing with {self.worker_count} workers..."
        )

        # Choose executor based on use_multiprocessing flag
        executor_class = (
            ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
        )

        try:
            with executor_class(max_workers=self.worker_count) as executor:
                # Process in batches
                for batch_samples in self.researcher_repo.batch_iterator(
                    batch_size=self.batch_size, status=ProcessingStatus.PENDING
                ):
                    # Submit tasks to executor
                    futures = [
                        executor.submit(self._process_sample_wrapper, sample, process_fn)
                        for sample in batch_samples
                    ]

                    # Wait for all tasks to complete
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Worker error: {e}")

                    # Update progress
                    self._update_progress()

            # Mark batch as completed
            self.batch_repo.update_status(
                self.batch_name, ProcessingStatus.COMPLETED
            )
            self._log_final_stats()

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.batch_repo.update_status(self.batch_name, ProcessingStatus.FAILED)
            raise

    def _process_sample_wrapper(
        self, sample: ResearcherRecord, process_fn: Callable
    ) -> None:
        """
        Wrapper for processing a single sample with error handling.

        Args:
            sample: Researcher sample to process
            process_fn: Function to process the sample
        """
        retries = 0

        while retries <= self.max_retries:
            try:
                # Mark as processing
                self.researcher_repo.update_status(
                    sample.researcher_id, ProcessingStatus.PROCESSING
                )

                # Process the sample
                process_fn(sample)

                # Mark as completed
                self.researcher_repo.update_status(
                    sample.researcher_id, ProcessingStatus.COMPLETED
                )

                self.successful_count += 1
                return

            except Exception as e:
                retries += 1
                logger.warning(
                    f"Error processing {sample.researcher_id} "
                    f"(attempt {retries}/{self.max_retries}): {e}"
                )

                if retries > self.max_retries:
                    # Mark as failed
                    self.researcher_repo.update_status(
                        sample.researcher_id, ProcessingStatus.FAILED
                    )
                    self.failed_count += 1
                    logger.error(
                        f"Failed to process {sample.researcher_id} after "
                        f"{self.max_retries} retries"
                    )
                    return

                # Exponential backoff
                time.sleep(2**retries)

    def _update_progress(self) -> None:
        """Update batch progress in database."""
        self.processed_count = self.successful_count + self.failed_count

        self.batch_repo.update_progress(
            self.batch_name,
            self.processed_count,
            self.successful_count,
            self.failed_count,
        )

        # Log progress
        if self.processed_count % 100 == 0:
            elapsed = time.time() - (self.start_time or time.time())
            rate = self.processed_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {self.processed_count} processed "
                f"({self.successful_count} success, {self.failed_count} failed) "
                f"- {rate:.2f} samples/sec"
            )

    def _log_final_stats(self) -> None:
        """Log final processing statistics."""
        elapsed = time.time() - (self.start_time or time.time())
        rate = self.processed_count / elapsed if elapsed > 0 else 0

        logger.info("=" * 70)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Batch name: {self.batch_name}")
        logger.info(f"Total processed: {self.processed_count}")
        logger.info(f"Successful: {self.successful_count}")
        logger.info(f"Failed: {self.failed_count}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info(f"Processing rate: {rate:.2f} samples/sec")
        logger.info("=" * 70)

    async def rate_limit(self) -> None:
        """
        Apply rate limiting for API calls.

        Ensures we don't exceed rate_limit_per_minute API calls.
        """
        async with self._rate_limit_lock:
            now = time.time()

            # Remove calls older than 1 minute
            self._api_call_times = [
                t for t in self._api_call_times if now - t < 60
            ]

            # Check if we need to wait
            if len(self._api_call_times) >= self.rate_limit_per_minute:
                # Wait until the oldest call expires
                sleep_time = 60 - (now - self._api_call_times[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, waiting {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)

                # Remove expired calls
                now = time.time()
                self._api_call_times = [
                    t for t in self._api_call_times if now - t < 60
                ]

            # Record this call
            self._api_call_times.append(time.time())

    def get_stats(self) -> dict:
        """
        Get current batch statistics.

        Returns:
            Dictionary with batch stats
        """
        batch = self.batch_repo.get_by_name(self.batch_name)
        if not batch:
            return {}

        elapsed = 0
        if batch.started_at:
            end_time = batch.completed_at or datetime.now()
            elapsed = (end_time - batch.started_at).total_seconds()

        rate = batch.processed_samples / elapsed if elapsed > 0 else 0

        return {
            "batch_name": batch.batch_name,
            "status": batch.status.value,
            "total_samples": batch.total_samples,
            "processed_samples": batch.processed_samples,
            "successful_samples": batch.successful_samples,
            "failed_samples": batch.failed_samples,
            "worker_count": batch.worker_count,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
            "progress_percentage": (
                batch.processed_samples / batch.total_samples * 100
                if batch.total_samples > 0
                else 0
            ),
        }

    def close(self) -> None:
        """Close database connections."""
        self.db.close()
