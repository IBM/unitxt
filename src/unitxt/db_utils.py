import glob
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, List, Optional

import requests
from huggingface_hub import snapshot_download
from requests.exceptions import ConnectionError, ReadTimeout

from .logging_utils import get_logger
from .types import SQLDatabase

logger = get_logger()


class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""

    def __init__(self, db_config: SQLDatabase):
        self.db_config = db_config
        self.databases_folder = os.path.join(
            os.environ.get("UNITXT_TEXT2SQL_CACHE", "cache/text2sql"), "databases"
        )
        os.makedirs(self.databases_folder, exist_ok=True)

    @abstractmethod
    def get_table_schema(
        self,
    ) -> str:
        """Abstract method to get database schema."""
        pass

    @abstractmethod
    def execute_query(self, query: str) -> Any:
        """Abstract method to execute a query against the database."""
        pass


@lru_cache(maxsize=128)
def execute_query_local(db_path: str, query: str) -> Any:
    """Executes a query against the SQLite database."""
    conn = None  # Initialize conn to None outside the try block
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.info(f"Error executing SQL: {e}")
        return None
    finally:
        if conn:
            conn.close()


class LocalSQLiteConnector(DatabaseConnector):
    """Database connector for SQLite databases."""

    def __init__(self, db_config: SQLDatabase):
        super().__init__(db_config)
        db_id = self.db_config.get("db_id")
        if not db_id:
            raise ValueError("db_id is required for SQLiteConnector.")
        self.db_path = self.get_db_file_path(db_id)
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        self.cursor: sqlite3.Cursor = self.conn.cursor()

    def download_database(self, db_id):
        """Downloads the database from huggingface if needed."""
        done_file_path = os.path.join(self.databases_folder, "download_done")
        if "bird/" in db_id:
            if not os.path.exists(done_file_path):
                snapshot_download(
                    repo_id="premai-io/birdbench",
                    repo_type="dataset",
                    local_dir=self.databases_folder,
                    force_download=False,
                    allow_patterns="*validation*",
                )
                open(os.path.join(self.databases_folder, "download_done"), "w").close()
        else:
            raise NotImplementedError(
                f"current local db: {db_id} is not supported, only bird"
            )

    def get_db_file_path(self, db_id):
        """Gets the local path of a downloaded database file."""
        self.download_database(db_id)
        db_id = db_id.split("/")[-1]

        db_file_pattern = os.path.join(self.databases_folder, "**", db_id + ".sqlite")
        db_file_paths = glob.glob(db_file_pattern, recursive=True)

        if not db_file_paths:
            raise FileNotFoundError(f"Database file {db_id} not found.")
        if len(db_file_paths) > 1:
            raise FileExistsError(f"More than one files matched for {db_id}")
        return db_file_paths[0]

    def get_table_schema(
        self,
    ) -> str:
        """Extracts schema from an SQLite database."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables: list[tuple[str]] = self.cursor.fetchall()
        schemas: dict[str, str] = {}

        for table in tables:
            if isinstance(table, tuple):
                table = table[0]
            if table == "sqlite_sequence":
                continue
            sql_query: str = (
                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';"
            )
            self.cursor.execute(sql_query)
            schema_prompt: str = self.cursor.fetchone()[0]

            schemas[table] = schema_prompt

        schema_prompt: str = "\n\n".join(list(schemas.values()))
        return schema_prompt

    def execute_query(self, query: str) -> Any:
        """Executes a query against the SQLite database."""
        return execute_query_local(self.db_path, query)


class InMemoryDatabaseConnector(DatabaseConnector):
    """Database connector for mocking databases with in-memory data structures."""

    def __init__(self, db_config: SQLDatabase):
        super().__init__(db_config)
        self.tables = db_config.get("data", None)

        if not self.tables:
            raise ValueError("data is required for InMemoryDatabaseConnector.")

    def get_table_schema(
        self,
        select_tables: Optional[List[str]] = None,
    ) -> str:
        """Generates a mock schema from the tables structure."""
        schemas = {}
        for table_name, table_data in self.tables.items():
            if select_tables and table_name.lower() not in select_tables:
                continue
            columns = ", ".join([f"`{col}` TEXT" for col in table_data["columns"]])
            schema = f"CREATE TABLE `{table_name}` ({columns});"

            schemas[table_name] = schema

        return "\n\n".join(list(schemas.values()))

    def execute_query(self, query: str) -> Any:
        """Simulates executing a query against the mock database."""
        # Initialize in-memory database from the 'tables' dictionary
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        logger.debug("Running SQL query over in-memory DB")

        # Create tables and insert data from the 'db' dictionary
        for table_name, table_data in self.tables.items():
            columns = table_data["columns"]
            rows = table_data["rows"]

            # Create table
            cursor.execute(f"CREATE TABLE {table_name} ({', '.join(columns)})")

            # Insert data
            placeholders = ", ".join(["?"] * len(columns))
            cursor.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})", rows
            )

        try:
            cursor.execute(query)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.info(f"Error executing SQL: {e}")
            return None
        finally:
            conn.close()


@lru_cache(maxsize=128)
def execute_query_remote(
    api_url: str,
    database_id: str,
    api_key: str,
    query: str,
    retryable_exceptions: tuple = (ConnectionError, ReadTimeout),
    max_retries: int = 3,
    retry_delay: int = 5,  # seconds
    timeout: int = 30,  # seconds
) -> Optional[dict]:
    """Executes a query against the remote database, with retries for certain exceptions."""
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    retries = 0
    while retries <= max_retries:
        try:
            response = requests.post(
                f"{api_url}/sql",
                headers=headers,
                json={"sql": query, "dataSourceId": database_id},
                verify=True,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()

        except retryable_exceptions as e:
            retries += 1
            logger.warning(
                f"Attempt {retries} failed with error: {e}. Retrying in {retry_delay} seconds."
            )
            if retries <= max_retries:
                time.sleep(retry_delay)
            else:
                logger.error(f"Max retries ({max_retries}) exceeded for query: {query}")
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 500:
                retries += 1
                logger.warning(
                    f"Server error, attempt {retries} failed with error: {e}. Retrying in {retry_delay} seconds."
                )
                if retries <= max_retries:
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Max retries ({max_retries}) exceeded for query: {query}"
                    )
                    return None
            else:
                logger.error(f"HTTP Error on attempt {retries}: {e}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error on attempt {retries}: {e}")
            return None

    return None


class RemoteDatabaseConnector(DatabaseConnector):
    """Database connector for remote databases accessed via HTTP."""

    def __init__(self, db_config: SQLDatabase):
        super().__init__(db_config)

        assert db_config[
            "db_id"
        ], "db_id must be in db_config for RemoteDatabaseConnector"
        self.api_url, self.database_id = (
            db_config["db_id"].split(",")[0],
            db_config["db_id"].split("db_id=")[-1].split(",")[0],
        )

        if not self.api_url or not self.database_id:
            raise ValueError(
                "Both 'api_url' and 'database_id' are required for RemoteDatabaseConnector."
            )

        self.api_key = os.getenv("SQL_API_KEY", None)
        if not self.api_key:
            raise ValueError(
                "The environment variable 'SQL_API_KEY' must be set to use the RemoteDatabaseConnector."
            )

        self.headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        self.timeout = 30

    def get_table_schema(
        self,
    ) -> str:
        """Retrieves the schema of a database."""
        cur_api_url = f"{self.api_url}/datasource/{self.database_id}"
        response = requests.get(
            cur_api_url,
            headers=self.headers,
            verify=True,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            schema = response.json()["schema"]
        else:
            raise OSError(f"Could not fetch schema from {cur_api_url}")

        schema_text = ""
        for table in schema["tables"]:
            schema_text += f"Table: {table['table_name']} has columns: {[col['column_name'] for col in table['columns']]}\n"

        return schema_text

    def execute_query(self, query: str) -> Any:
        """Executes a query against the remote database, with retries for certain exceptions."""
        return execute_query_remote(
            api_url=self.api_url,
            database_id=self.database_id,
            api_key=self.api_key,
            query=query,
            timeout=self.timeout,
        )


def get_db_connector(db_type: str):
    """Creates and returns the appropriate DatabaseConnector instance based on db_type."""
    if db_type == "local":
        connector = LocalSQLiteConnector
    elif db_type == "in_memory":
        connector = InMemoryDatabaseConnector
    elif db_type == "remote":
        connector = RemoteDatabaseConnector

    else:
        raise ValueError(f"Unsupported database type: {db_type}")

    return connector
