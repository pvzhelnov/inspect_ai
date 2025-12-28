"""
Database manager for researcher profiling production system.

Handles SQLite database connections, transactions, and queries with
connection pooling for high concurrency and millions of records.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    """Database configuration."""

    db_path: Path
    pool_size: int = 10
    timeout: float = 30.0
    check_same_thread: bool = False
    enable_wal: bool = True  # Write-Ahead Logging for better concurrency
    cache_size: int = -64000  # 64MB cache (negative = KB)
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"  # Faster than FULL, still safe with WAL


class ConnectionPool:
    """Thread-safe connection pool for SQLite."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._local = threading.local()
        self._lock = threading.Lock()
        self._pool: list[sqlite3.Connection] = []
        self._in_use: set[sqlite3.Connection] = set()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(
            str(self.config.db_path),
            timeout=self.config.timeout,
            check_same_thread=self.config.check_same_thread,
        )

        # Enable WAL mode for better concurrency
        if self.config.enable_wal:
            conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            conn.execute(f"PRAGMA synchronous = {self.config.synchronous}")

        # Set cache size
        conn.execute(f"PRAGMA cache_size = {self.config.cache_size}")

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Use row factory for dict-like access
        conn.row_factory = sqlite3.Row

        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        # Check if thread has a connection
        if hasattr(self._local, "connection") and self._local.connection:
            return self._local.connection

        with self._lock:
            if self._pool:
                conn = self._pool.pop()
            else:
                conn = self._create_connection()

            self._in_use.add(conn)
            self._local.connection = conn
            return conn

    def release_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)

            # Rollback any uncommitted transactions
            try:
                conn.rollback()
            except Exception:
                pass

            # Return to pool if under limit
            if len(self._pool) < self.config.pool_size:
                self._pool.append(conn)
            else:
                conn.close()

            # Clear thread-local connection
            if hasattr(self._local, "connection"):
                self._local.connection = None

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass

            for conn in self._in_use:
                try:
                    conn.close()
                except Exception:
                    pass

            self._pool.clear()
            self._in_use.clear()


class Database:
    """Database manager with connection pooling and transaction support."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = ConnectionPool(config)
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database schema if not exists."""
        # Create database directory if needed
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Load and execute schema
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            with open(schema_path) as f:
                schema_sql = f.read()

            with self.get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a connection from the pool (context manager)."""
        conn = self.pool.get_connection()
        try:
            yield conn
        finally:
            self.pool.release_connection(conn)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Execute operations in a transaction."""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def execute(
        self, sql: str, params: tuple | dict | None = None
    ) -> sqlite3.Cursor:
        """Execute a single SQL statement."""
        with self.get_connection() as conn:
            if params:
                return conn.execute(sql, params)
            return conn.execute(sql)

    def executemany(
        self, sql: str, params_list: list[tuple | dict]
    ) -> sqlite3.Cursor:
        """Execute SQL statement with multiple parameter sets."""
        with self.transaction() as conn:
            return conn.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple | dict | None = None) -> Optional[dict]:
        """Fetch a single row as dict."""
        cursor = self.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetchall(
        self, sql: str, params: tuple | dict | None = None
    ) -> list[dict]:
        """Fetch all rows as list of dicts."""
        cursor = self.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def fetchmany(
        self, sql: str, size: int, params: tuple | dict | None = None
    ) -> list[dict]:
        """Fetch many rows as list of dicts."""
        cursor = self.execute(sql, params)
        return [dict(row) for row in cursor.fetchmany(size)]

    def insert(self, table: str, data: dict) -> int:
        """Insert a row and return the row ID."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        with self.transaction() as conn:
            cursor = conn.execute(sql, data)
            return cursor.lastrowid

    def insert_many(self, table: str, data_list: list[dict]) -> int:
        """Insert multiple rows and return count."""
        if not data_list:
            return 0

        columns = ", ".join(data_list[0].keys())
        placeholders = ", ".join(f":{k}" for k in data_list[0].keys())
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        with self.transaction() as conn:
            cursor = conn.executemany(sql, data_list)
            return cursor.rowcount

    def update(
        self, table: str, data: dict, where: str, where_params: tuple | dict
    ) -> int:
        """Update rows and return count."""
        set_clause = ", ".join(f"{k} = :{k}" for k in data.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"

        # Merge data and where_params
        if isinstance(where_params, dict):
            params = {**data, **where_params}
        else:
            params = data

        with self.transaction() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    def delete(self, table: str, where: str, where_params: tuple | dict) -> int:
        """Delete rows and return count."""
        sql = f"DELETE FROM {table} WHERE {where}"

        with self.transaction() as conn:
            cursor = conn.execute(sql, where_params)
            return cursor.rowcount

    def count(self, table: str, where: str = "", where_params: tuple | dict | None = None) -> int:
        """Count rows matching condition."""
        sql = f"SELECT COUNT(*) as count FROM {table}"
        if where:
            sql += f" WHERE {where}"

        result = self.fetchone(sql, where_params)
        return result["count"] if result else 0

    def batch_iterator(
        self, sql: str, batch_size: int = 1000, params: tuple | dict | None = None
    ) -> Iterator[list[dict]]:
        """Iterate over query results in batches."""
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params) if params else conn.execute(sql)

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield [dict(row) for row in rows]

    def close(self) -> None:
        """Close all connections."""
        self.pool.close_all()

    # Utility methods for JSON fields
    @staticmethod
    def to_json(data: Any) -> str:
        """Convert Python object to JSON string."""
        return json.dumps(data) if data is not None else None

    @staticmethod
    def from_json(data: str | None) -> Any:
        """Convert JSON string to Python object."""
        return json.loads(data) if data else None
