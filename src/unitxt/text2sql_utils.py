import functools
import glob
import hashlib
import json
import os
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from huggingface_hub import snapshot_download
from requests.exceptions import ConnectionError, ReadTimeout

from .logging_utils import get_logger
from .types import SQLDatabase

logger = get_logger()

# Check if caching is enabled via environment variable
CACHE_LOCATION = os.getenv("UNITXT_CACHE_LOCATION")

# Set max cache size to 10GB or the value of env var MAX_CACHE_SIZE
MAX_CACHE_SIZE = os.getenv("MAX_CACHE_SIZE", 10 * 1024**3)

_cache_instance = None


class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""

    def __init__(self, db_config: SQLDatabase):
        self.db_config = db_config
        self.databases_folder = os.path.join(
            os.environ.get("UNITXT_CACHE_LOCATION", "cache/text2sql"), "databases"
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
        return cursor.fetchall(), None
    except sqlite3.Error as e:
        logger.info(f"Error executing SQL: {e}")
        return None, f"Error executing SQL: {e}"
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
            return cursor.fetchall(), None
        except sqlite3.Error as e:
            logger.info(f"Error executing SQL: {e}")
            return None, f"Error executing SQL: {e}"
        finally:
            conn.close()


def get_cache():
    """Returns a singleton cache instance, initializing it if necessary."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = Cache()
    return _cache_instance


def generate_cache_key(*args, **kwargs):
    """Generate a stable hashable cache key for various input types.

    :param args: Positional arguments of the function.
    :param kwargs: Keyword arguments of the function.
    :return: A hashed key as a string.
    """
    try:
        # Convert args and kwargs to a JSON string (sorted to ensure consistency)
        serialized = json.dumps(
            {"args": args, "kwargs": kwargs}, sort_keys=True, default=str
        )
    except TypeError:
        # Fallback for non-serializable objects
        serialized = repr((args, kwargs))

    # Hash the serialized data
    return hashlib.md5(serialized.encode()).hexdigest()


class Cache:
    """A class that provides disk-based caching functionality for a given function."""

    def __init__(self):
        """Initializes the cache.

        If `CACHE_LOCATION` (os.getenv("UNITXT_CACHE_LOCATION") is set, a disk-based
        cache is created using `diskcache`.

        Args:
            None

        Returns:
            None
        """
        if CACHE_LOCATION:
            try:
                import diskcache

                # Ensure the cache directory exists
                os.makedirs(CACHE_LOCATION, exist_ok=True)

                # Create a global diskcache Cache instance
                self.cache = diskcache.Cache(CACHE_LOCATION, size_limit=MAX_CACHE_SIZE)
                logger.info(f"Caching enabled at {CACHE_LOCATION}")
            except ImportError as e:
                raise ImportError(
                    "UNITXT_CACHE_LOCATION is set, but diskcache is not installed.\n"
                    "Please install diskcache `pip install diskcache` "
                    "or unset UNITXT_CACHE_LOCATION."
                ) from e
        else:
            self.cache = None  # Disable caching

    def get_or_set(self, key, compute_fn, no_cache=False, refresh=False):
        if not self.cache or no_cache:
            logger.info(f"Bypassing cache for key: {key}")
            return compute_fn()

        if refresh and key in self.cache:
            logger.info(f"Refreshing cache for key: {key}")
            del self.cache[key]

        if key in self.cache:
            logger.info(f"Cache hit for key: {key}")
            return self.cache[key]

        logger.info(f"Cache miss for key: {key}. Computing value...")
        result = compute_fn()

        if result and not (
            isinstance(result, tuple) and len(result) == 2 and result[0] is None
        ):
            self.cache[key] = result
            logger.info(f"Stored result in cache for key: {key}")
        else:
            logger.info(f"None result. Bypassing caching for key: {key}")

        return result

    async def async_get_or_set(self, key, compute_fn, no_cache=False, refresh=False):
        if not self.cache or no_cache:
            logger.info(f"Bypassing cache for key: {key}")
            return await compute_fn()

        if refresh and key in self.cache:
            logger.info(f"Refreshing cache for key: {key}")
            del self.cache[key]

        if key in self.cache:
            logger.info(f"Cache hit for key: {key}")
            return self.cache[key]

        logger.info(f"Cache miss for key: {key}. Computing value asynchronously...")
        result = await compute_fn()
        self.cache[key] = result
        logger.info(f"Stored result in cache for key: {key}")
        return result

    def memoize(self, key_func=generate_cache_key, no_cache=False, refresh=False):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.cache or no_cache:
                    logger.info(f"Bypassing cache for function: {func.__name__}")
                    return func(*args, **kwargs)

                key = key_func(func.__name__, *args, **kwargs)

                if refresh and key in self.cache:
                    logger.info(
                        f"Refreshing cache for function: {func.__name__}, key: {key}"
                    )
                    del self.cache[key]

                if key in self.cache:
                    logger.info(f"Cache hit for function: {func.__name__}, key: {key}")
                    return self.cache[key]

                logger.info(
                    f"Cache miss for function: {func.__name__}, key: {key}. Computing value..."
                )
                result = func(*args, **kwargs)
                self.cache[key] = result
                logger.info(
                    f"Stored result in cache for function: {func.__name__}, key: {key}"
                )
                return result

            return wrapper

        return decorator

    def async_memoize(self, key_func=generate_cache_key, no_cache=False, refresh=False):
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if no_cache:
                    logger.info(f"Bypassing cache for async function: {func.__name__}")
                    return await func(*args, **kwargs)

                key = key_func(func.__name__, *args, **kwargs)

                if refresh and key in self.cache:
                    logger.info(
                        f"Refreshing cache for async function: {func.__name__}, key: {key}"
                    )
                    del self.cache[key]

                if key in self.cache:
                    logger.info(
                        f"Cache hit for async function: {func.__name__}, key: {key}"
                    )
                    return self.cache[key]

                logger.info(
                    f"Cache miss for async function: {func.__name__}, key: {key}. Computing value..."
                )
                result = await func(*args, **kwargs)
                self.cache[key] = result
                logger.info(
                    f"Stored result in cache for async function: {func.__name__}, key: {key}"
                )
                return result

            return wrapper

        return decorator


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
) -> (Optional[dict], str):
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
                verify=False,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json(), None

        except retryable_exceptions as e:
            retries += 1
            logger.warning(
                f"Attempt {retries} failed with error: {e}. Retrying in {retry_delay} seconds."
            )
            if retries <= max_retries:
                time.sleep(retry_delay)
            else:
                logger.error(f"Max retries ({max_retries}) exceeded for query: {query}")
                return (
                    None,
                    f"Max retries ({max_retries}) exceeded for query: {query} - Error: {e!s}",
                )

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
                    return (
                        None,
                        f"Max retries ({max_retries}) exceeded for query: {query} - Error: {e!s}",
                    )
            else:
                logger.error(f"HTTP Error on attempt {retries}: {e}")
                return (
                    None,
                    f"HTTP Error on attempt {retries}: {e}",
                )

        except Exception as e:
            logger.error(f"Unexpected error on attempt {retries}: {e}")
            return (None, f"Unexpected error on attempt {retries}: {e}")

    return None, "Unknown Error in SQL execution"


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
        cur_api_url = f"{self.api_url}/datasources/{self.database_id}"
        response = requests.get(
            cur_api_url,
            headers=self.headers,
            verify=False,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            schema = response.json()["schema"]
        else:
            raise OSError(f"Could not fetch schema from {cur_api_url}")

        schema_text = ""
        for table in schema["tables"]:
            schema_text += f"Table: {table['name'] if 'name' in table else table['table_name']} has columns: {[col['name'] if 'name' in col else col['column_name'] for col in table['columns']]}\n"

        return schema_text

    def execute_query(self, query: str) -> Any:
        """Executes a query against the remote database, with retries for certain exceptions."""
        cache = get_cache()

        cache_key = generate_cache_key(
            "sql_request", self.api_url, self.database_id, query
        )
        return cache.get_or_set(
            cache_key,
            lambda: execute_query_remote(
                api_url=self.api_url,
                database_id=self.database_id,
                api_key=self.api_key,
                query=query,
                timeout=self.timeout,
            ),
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


@dataclass
class SQLNonExecutionMetricResult:
    sqlglot_validity: int  # Whether SQL parses with sqlglot
    sqlparse_validity: int  # Whether SQL parses with sqlparse
    sqlglot_equivalence: int  # Semantic equivalence using sqlglot AST
    sqlglot_optimized_equivalence: int  # Equivalence after optimization via sqlglot
    sqlparse_equivalence: int  # Equivalence using sqlparse AST
    sql_exact_match: int  # Exact string match of predicted and gold SQL
    sql_syntactic_equivalence: int  # Any of the above equivalence conditions hold


def is_sqlglot_parsable(sql: str, db_type="sqlite") -> bool:
    """Returns True if sqlglot does not encounter any error, False otherwise."""
    from sqlglot import parse

    if not sql.strip():
        return False
    if db_type == "db2":
        db_type = "postgres"  ## TODO: temporary until sqlglot adds support for db2
    try:
        parse(sql, read=db_type)
        return True
    except Exception as e:
        logger.debug(f"SQL query could not parse: {e}")
        return False


def is_sqlparse_parsable(sql: str) -> bool:
    """Returns True if sqlparse does not encounter any error, False otherwise."""
    from sqlparse import parse
    from sqlparse.tokens import Error

    if not sql.strip():
        return False
    try:
        statements = parse(sql)
        for statement in statements:
            for token in statement.tokens:
                if token.ttype == Error:
                    return False
        return True
    except Exception as e:
        logger.debug(f"SQL query could not parse: {e}")
        return False


def sqlglot_optimized_equivalence(expected: str, generated: str) -> int:
    """Checks if SQL queries are equivalent using SQLGlot parsing, so we don't run them."""
    from sqlglot import diff, parse_one
    from sqlglot.optimizer import optimize

    try:
        t_diff = diff(
            optimize(parse_one(expected.lower()).sql(pretty=True)),
            optimize(parse_one(generated.lower()).sql(pretty=True)),
        )
        sql_diff = sum(0 if (e.__class__.__name__ == "Keep") else 1 for e in t_diff)

        return 1 if sql_diff == 0 else 0
    except Exception as e:
        logger.debug(f"Error parsing SQL for comparison: {e}")
        return False


def extract_select_columns(statement):
    """Parse SQL using sqlparse and extract columns."""
    from sqlparse.sql import Identifier, IdentifierList
    from sqlparse.tokens import DML, Keyword

    columns = []
    select_seen = False
    for token in statement.tokens:
        if token.ttype is DML and token.value.upper() == "SELECT":
            select_seen = True
            continue
        if select_seen:
            if token.ttype is Keyword and token.value.upper() in (
                "FROM",
                "WHERE",
                "GROUP",
                "HAVING",
                "ORDER",
                "LIMIT",
            ):
                break
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    columns.append(strip_alias(identifier.value))
            elif isinstance(token, Identifier):
                columns.append(strip_alias(token.value))
            else:
                val = token.value.strip()
                if val:
                    columns.append(strip_alias(val))
    return frozenset(columns)


def strip_alias(col: str) -> str:
    """Remove any AS alias from a column."""
    col = col.strip()
    upper = col.upper()
    if " AS " in upper:
        return col[: upper.index(" AS ")].strip()
    parts_alias = col.split()
    if len(parts_alias) > 1:
        return " ".join(parts_alias[:-1])
    return col


def collect_clause(statement, clause_keyword):
    """Parse SQL statement and collect clauses."""
    from sqlparse.tokens import Keyword

    found = False
    collected = []
    for token in statement.tokens:
        tvalue = token.value.upper()
        if token.ttype is Keyword:
            if tvalue.startswith(clause_keyword):
                found = True
                continue
            if found and tvalue in (
                "FROM",
                "WHERE",
                "GROUP",
                "HAVING",
                "ORDER",
                "LIMIT",
            ):
                break
        if found:
            collected.append(token.value)
    return " ".join(collected).strip()


def extract_select_info(sql: str):
    """Parse SQL using sqlparse and return a dict of extracted columns and clauses."""
    from sqlparse import parse
    from sqlparse.tokens import DML

    statements = parse(sql)
    if len(statements) != 1:
        return None
    stmt = statements[0]
    if not any(t.ttype is DML and t.value.upper() == "SELECT" for t in stmt.tokens):
        return None
    parts = {
        "columns": None,
        "from": "",
        "where": "",
        "group": "",
        "having": "",
        "order": "",
    }
    columns = extract_select_columns(stmt)
    if not columns:
        columns = frozenset()
    parts["columns"] = columns
    parts["from"] = collect_clause(stmt, "FROM")
    parts["where"] = collect_clause(stmt, "WHERE")
    parts["group"] = collect_clause(stmt, "GROUP")
    parts["having"] = collect_clause(stmt, "HAVING")
    parts["order"] = collect_clause(stmt, "ORDER")
    return parts


def sqlparse_queries_equivalent(sql1: str, sql2: str) -> bool:
    """Returns True if both SQL queries are naively considered equivalent."""
    try:
        info1 = extract_select_info(sql1)
        info2 = extract_select_info(sql2)
        if not info1 or not info2:
            return False
        if info1["columns"] != info2["columns"]:
            return False
        for k in ["from", "where", "group", "having", "order"]:
            if info1[k].replace(" ", "").upper() != info2[k].replace(" ", "").upper():
                return False
        return True
    except Exception as e:
        logger.debug(f"Errpr parsing SQL query for comparison: {e}")
        return False


def sqlglot_parsed_queries_equivalent(sql1: str, sql2: str, dialect: str = "") -> bool:
    """Return True if two SQL queries match after parsing with SQLGlot."""
    from sqlglot import exp, parse_one

    try:
        ast1 = parse_one(sql1, read=dialect)
        ast2 = parse_one(sql2, read=dialect)
    except:
        return False
    if not (isinstance(ast1, exp.Select) and isinstance(ast2, exp.Select)):
        return False

    def normalized_select_columns(select_expr: exp.Select):
        cols = []
        for item in select_expr.expressions:
            copy_item = item.copy()
            copy_item.set("alias", None)
            cols.append(copy_item.sql(dialect=dialect, normalize=True))
        return frozenset(cols)

    if normalized_select_columns(ast1) != normalized_select_columns(ast2):
        return False

    def normalized_clause(expr: exp.Expression, key: str):
        clause = expr.args.get(key)
        return clause.sql(dialect=dialect, normalize=True) if clause else ""

    for clause_key in ("from", "where", "group", "having", "order"):
        if normalized_clause(ast1, clause_key) != normalized_clause(ast2, clause_key):
            return False

    return True


def sql_exact_match(sql1: str, sql2: str) -> bool:
    """Return True if two SQL strings match after very basic normalization."""

    def normalize_sql(s: str) -> str:
        s = s.strip().rstrip(";")
        s = re.sub(r"\s+", " ", s)
        return s.upper()

    return normalize_sql(sql1) == normalize_sql(sql2)


@dataclass
class SQLExecutionResult:
    execution_accuracy: int  # Whether the predicted and gold SQL results match exactly
    non_empty_execution_accuracy: (
        int  # Same as execution_accuracy but only if gold is non-empty
    )
    subset_non_empty_execution_accuracy: (
        int  # Whether predicted is a subset of gold or vice versa, non-empty only
    )
    execution_accuracy_bird: (
        int  # Whether the predicted SQL matches gold using BIRD evaluation logic
    )
    non_empty_gold_df: int  # Whether the gold SQL produced a non-empty dataframe
    gold_sql_runtime: float  # Time taken to execute the gold SQL
    predicted_sql_runtime: float  # Time taken to execute the predicted SQL
    pred_to_gold_runtime_ratio: float  # Ratio of predicted runtime to gold runtime
    gold_error: int  # Whether the gold SQL had an execution error
    predicted_error: int  # Whether the predicted SQL had an execution error
    gold_df_json: str  # JSON representation of the gold SQL result dataframe
    predicted_df_json: str  # JSON representation of the predicted SQL result dataframe
    error_message: str  # Error message from predicted execution if any


def compare_dfs_ignore_colnames_ordered_rows(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> bool:
    if df1.shape != df2.shape:
        return False
    df1_sorted_rows = np.array([np.sort(row) for row in df1.values.astype(str)])
    df2_sorted_rows = np.array([np.sort(row) for row in df2.values.astype(str)])
    return np.array_equal(df1_sorted_rows, df2_sorted_rows)


def compare_dfs_ignore_colnames_unordered_rows(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> bool:
    if df1.shape != df2.shape:
        return False
    df1_sorted = np.sort(np.sort(df1.values.astype(str), axis=1), axis=0)
    df2_sorted = np.sort(np.sort(df2.values.astype(str), axis=1), axis=0)
    return np.array_equal(df1_sorted, df2_sorted)


def compare_dfs_ignore_colnames_subset(
    df1: pd.DataFrame, df2: pd.DataFrame, ignore_row_order: bool = True
) -> bool:
    """Checks if the smaller of the two DataFrames is likely a subset of the other.

    Subset comparison is column-based, to support Text2SQL evaluation for when the
    predicted SQL dataframe has missing or additional columns. Each row is treated as
    a multiset of (stringified) values, and the function checks if every row in the
    smaller DataFrame (by column count) is a multiset subset of the corresponding row
    in the larger DataFrame. When ground truth SQL does not have ORDER BY,
    ignore_row_order can be set to True to ignore the order of rows. In this case,
    column values are sorted before comparison. This means that there could be cases
    where the dataframes have the exact same number of rows and column values after
    sort are the same, but the dataframes are not actually a subset of each other.
    This is unlikely to happen in practice, but the score is not guaranteed to be
    100% accurate and may overestimate the accuracy.

    Args:
        df1 (pd.DataFrame): The first DataFrame to compare.
        df2 (pd.DataFrame): The second DataFrame to compare.
        ignore_row_order (bool, optional): If True, ignores the order of rows by
            sorting them before comparison. Defaults to True.

    Returns:
        bool: True if the smaller DataFrame (column-wise) is likely a subset of the
            larger one, False otherwise.
    """

    def row_to_multiset(row):
        return Counter(str(x) for x in row)

    def rows_to_multisets(df):
        return [row_to_multiset(row) for row in df.values]

    def sort_df(df):
        sorted_df = df.copy()
        for col in sorted_df.columns:
            sorted_df[col] = sorted_df[col].astype(str).sort_values(ignore_index=True)
        return sorted_df

    if df1.empty or df2.empty or len(df1) != len(df2):
        return False

    subset_df, superset_df = (df1, df2) if df1.shape[1] <= df2.shape[1] else (df2, df1)

    if ignore_row_order:
        subset_df = sort_df(subset_df)
        superset_df = sort_df(superset_df)

    subset_rows = rows_to_multisets(subset_df)
    superset_rows = rows_to_multisets(superset_df)

    for r1, r2 in zip(subset_rows, superset_rows):
        if not all(r1[k] <= r2.get(k, 0) for k in r1):
            return False
    return True


def compare_dfs_bird_eval_logic(df1: pd.DataFrame, df2: pd.DataFrame):
    """Check if two SQL query result sets are exactly equal, as in BIRD evaluation.

    This function checks if the set of rows returned by the predicted SQL query
    (`predicted_res`) is exactly equal to the set of rows returned by the ground truth
    SQL query (`ground_truth_res`). This is the logic used in the original BIRD
    evaluation code:
    https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py.
    """
    df1_set = {tuple(row) for row in df1.values.astype(str)}
    df2_set = {tuple(row) for row in df2.values.astype(str)}
    return int(df1_set == df2_set)


def compare_result_dfs(
    gold_df: pd.DataFrame, pred_df: pd.DataFrame, gold_sql: str
) -> Tuple[int, int, int]:
    """Compares two DataFrames representing SQL query results.

    Args:
        gold_df (pd.DataFrame): The ground truth DataFrame.
        pred_df (pd.DataFrame): The predicted DataFrame.
        gold_sql (str): The ground truth SQL query string.

    Returns:
        Tuple[int, int, int]: A tuple containing:
            - match (int): 1 if the predicted DataFrame matches the gold DataFrame
            - non_empty_match (int): 1 if both DataFrames are non-empty and match,
              0 otherwise.
            - subset_match (int): 1 if the predicted DataFrame is a subset or
              superset of the gold DataFrame.

    Notes:
        - The comparison ignores column names.
        - Row order is considered only if 'ORDER BY' is present in the SQL query.
    """
    subset_match = 0
    non_empty_match = 0
    if "ORDER BY" in gold_sql.upper():
        match = int(compare_dfs_ignore_colnames_ordered_rows(pred_df, gold_df))
        if not gold_df.empty and not pred_df.empty:
            non_empty_match = match
            if compare_dfs_ignore_colnames_subset(
                gold_df, pred_df, ignore_row_order=False
            ):
                subset_match = 1
    else:
        match = int(compare_dfs_ignore_colnames_unordered_rows(pred_df, gold_df))
        if not gold_df.empty and not pred_df.empty:
            non_empty_match = match
            if compare_dfs_ignore_colnames_subset(
                gold_df, pred_df, ignore_row_order=True
            ):
                subset_match = 1
    return match, non_empty_match, subset_match


def run_query(
    sql: str, connector, sql_timeout: float
) -> Tuple[Optional[pd.DataFrame], float, str]:
    """Executes a SQL query using the provided connector with a timeout.

    Args:
        sql (str): The SQL query string to execute.
        connector: An object with an `execute_query` method that executes the SQL
            query.
        sql_timeout (float): The maximum time in seconds to allow for query
            execution.

    Returns:
        Tuple[Optional[pd.DataFrame], float, str]:
            - A pandas DataFrame containing the query results, or None if an error
              occurred.
            - The duration in seconds taken to execute the query. 0.0 if an error.
            - An error message string if an error occurred, otherwise an empty
              string.

    Notes:
        - If the SQL string is empty or only whitespace, returns immediately with a
          message.
        - If the query execution exceeds the timeout, returns a timeout error
          message.
        - Any other exceptions are caught and returned as error messages.
    """
    import time

    from func_timeout import func_timeout
    from func_timeout.exceptions import FunctionTimedOut

    if not sql.strip():
        return None, 0.0, "No SQL query found in the prediction."

    try:
        start = time.perf_counter()
        result, error = func_timeout(sql_timeout, connector.execute_query, args=(sql,))
        duration = time.perf_counter() - start
        if isinstance(result, dict) and "results" in result:
            result = result["results"]
        if error:
            return None, duration, error
        return pd.DataFrame(result), duration, ""
    except FunctionTimedOut as e:
        return None, 0.0, f"Timeout: {e}"
    except Exception as e:
        return None, 0.0, f"Error: {e}"


def get_sql_execution_results(
    predicted_sql: str, gold_sql: str, connector, sql_timeout: float
) -> SQLExecutionResult:
    """Execute and compare predicted and gold SQL queries, returning execution metrics.

    Args:
        predicted_sql (str): The SQL query predicted by the model.
        gold_sql (str): The reference (gold) SQL query.
        connector: Database connector object used to execute the queries.
        sql_timeout (float): Maximum time (in seconds) allowed for query execution.

    Returns:
        SQLExecutionResult: An object containing various execution metrics, including:
            - execution_accuracy (int): 1 if predicted and gold queries produce
              equivalent results, else 0.
            - non_empty_execution_accuracy (int): 1 if both queries produce non-empty
              and equivalent results, else 0.
            - subset_non_empty_execution_accuracy (int): 1 if predicted results are a
              subset or superset of gold results and non-empty, else 0. Subset
              comparison is column-based. This means that the predicted SQL dataframe
              can have missing or additional columns compared to the gold SQL dataframe.
            - execution_accuracy_bird (int): 1 if results match according to BIRD
              evaluation logic, else 0.
            - non_empty_gold_df (int): 1 if the gold query result is non-empty, else 0.
            - gold_sql_runtime (float): Execution time for the gold SQL query.
            - predicted_sql_runtime (float): Execution time for the predicted SQL query.
            - pred_to_gold_runtime_ratio (float): Ratio of predicted to gold query
              runtimes.
            - gold_error (int): 1 if the gold query failed, else 0.
            - predicted_error (int): 1 if the predicted query failed, else 0.
            - gold_df_json (str): JSON representation of the gold query result
              DataFrame.
            - predicted_df_json (str): JSON representation of the predicted query
              result DataFrame.
            - error_message (str): Error message if any query failed, else empty
              string.

    Notes:
        - If the gold query fails, the function returns early with error details.
        - If the predicted query is identical or SQL-equivalent to the gold query,
          results are considered correct without execution.
        - Otherwise, both queries are executed and their results compared using
          multiple metrics.
    """
    gold_df, gold_runtime, gold_error_msg = run_query(gold_sql, connector, sql_timeout)
    gold_error = int(bool(gold_error_msg))

    if gold_error:
        return SQLExecutionResult(
            execution_accuracy=0,
            non_empty_execution_accuracy=0,
            subset_non_empty_execution_accuracy=0,
            execution_accuracy_bird=0,
            non_empty_gold_df=0,
            gold_sql_runtime=gold_runtime,
            predicted_sql_runtime=0,
            pred_to_gold_runtime_ratio=0,
            gold_error=gold_error,
            predicted_error=0,
            gold_df_json="",
            predicted_df_json="",
            error_message=gold_error_msg,
        )

    non_empty_gold_df = int(not gold_df.empty)
    if predicted_sql.strip().lower() == gold_sql.strip().lower():
        return SQLExecutionResult(
            execution_accuracy=1,
            non_empty_execution_accuracy=non_empty_gold_df,
            subset_non_empty_execution_accuracy=non_empty_gold_df,
            execution_accuracy_bird=1,
            non_empty_gold_df=non_empty_gold_df,
            gold_sql_runtime=gold_runtime,
            predicted_sql_runtime=0,
            pred_to_gold_runtime_ratio=0,
            gold_error=0,
            predicted_error=0,
            gold_df_json=gold_df.to_json(),
            predicted_df_json=gold_df.to_json(),
            error_message="",
        )

    try:
        if sqlglot_optimized_equivalence(gold_sql, predicted_sql):
            return SQLExecutionResult(
                execution_accuracy=1,
                non_empty_execution_accuracy=non_empty_gold_df,
                subset_non_empty_execution_accuracy=non_empty_gold_df,
                execution_accuracy_bird=1,
                non_empty_gold_df=non_empty_gold_df,
                gold_sql_runtime=gold_runtime,
                predicted_sql_runtime=0,
                pred_to_gold_runtime_ratio=0,
                gold_error=0,
                predicted_error=0,
                gold_df_json=gold_df.to_json(),
                predicted_df_json=gold_df.to_json(),
                error_message="",
            )
    except Exception as e:
        logger.info(f"Could not check SQL equivalence: {e}")

    pred_df, pred_runtime, pred_error_msg = run_query(
        predicted_sql, connector, sql_timeout
    )
    pred_error = 1 if pred_error_msg else 0

    if pred_df is None:
        return SQLExecutionResult(
            execution_accuracy=0,
            non_empty_execution_accuracy=0,
            subset_non_empty_execution_accuracy=0,
            execution_accuracy_bird=0,
            non_empty_gold_df=non_empty_gold_df,
            gold_sql_runtime=gold_runtime,
            predicted_sql_runtime=pred_runtime,
            pred_to_gold_runtime_ratio=(pred_runtime / gold_runtime)
            if gold_runtime > 0
            else 0,
            gold_error=0,
            predicted_error=pred_error,
            gold_df_json=gold_df.to_json(),
            predicted_df_json="",
            error_message=pred_error_msg,
        )

    match, non_empty_match, subset_match = compare_result_dfs(
        gold_df, pred_df, gold_sql
    )
    bird_match = compare_dfs_bird_eval_logic(gold_df, pred_df)

    return SQLExecutionResult(
        execution_accuracy=match,
        non_empty_execution_accuracy=non_empty_match,
        subset_non_empty_execution_accuracy=subset_match,
        execution_accuracy_bird=bird_match,
        non_empty_gold_df=non_empty_gold_df,
        gold_sql_runtime=gold_runtime,
        predicted_sql_runtime=pred_runtime,
        pred_to_gold_runtime_ratio=(pred_runtime / gold_runtime)
        if gold_runtime > 0
        else 0,
        gold_error=0,
        predicted_error=0,
        gold_df_json=gold_df.to_json(),
        predicted_df_json=pred_df.to_json(),
        error_message=pred_error_msg,
    )


def replace_select_clause(
    source_query: str, target_query: str, dialect: str = "postgres"
) -> str:
    """Replaces the SELECT clause of the target SQL query with the SELECT clause from the source SQL query.

    Args:
        source_query (str): SQL query whose SELECT clause will be used.
        target_query (str): SQL query whose SELECT clause will be replaced.
        dialect (str): SQL dialect for parsing and rendering (default: "postgres").

    Returns:
        str: A new SQL query with the SELECT clause of `target_query` replaced by that of `source_query`.

    Raises:
        ValueError: If either query is not a valid SELECT statement.

    Example:
        >>> replace_select_clause(
        ...     "SELECT id FROM employees",
        ...     "SELECT name FROM employees WHERE age > 30"
        ... )
        "SELECT id FROM employees WHERE age > 30"
    """
    from sqlglot import exp, parse_one

    if not dialect:
        dialect = "postgres"

    # Parse queries using the specified dialect
    source_ast = parse_one(source_query, read=dialect)
    target_ast = parse_one(target_query, read=dialect)

    if not isinstance(source_ast, exp.Select) or not isinstance(target_ast, exp.Select):
        raise ValueError("Both queries must be valid SELECT statements.")

    # Replace SELECT expressions in the target with those from the source
    target_ast.set("expressions", source_ast.expressions)

    # Return the updated SQL string using the dialect
    return target_ast.sql(dialect=dialect)


def extract_sql_from_text(text: str) -> str:
    """Extracts the first SQL query from the given text.

    Priority:
    1. SQL inside fenced blocks like ```sql ... ```
    2. SQL starting on a new line or after a colon/label
    3. SQL without semicolon

    Returns:
        The SQL query string, or an empty string if not found.
    """
    # 1. Look for fenced SQL code block
    fenced_block_pattern = re.compile(r"```sql\s+(.*?)```", re.IGNORECASE | re.DOTALL)
    match = fenced_block_pattern.search(text)
    if match:
        return match.group(1).strip()

    # 2. Inline SQL with semicolon
    sql_keywords = r"(?:SELECT|INSERT|UPDATE|DELETE|WITH)\s+"
    sql_start = (
        r"(?:^|\n|:\s*)"  # Start of string, newline, or colon label like "Just run:"
    )
    sql_pattern = re.compile(
        rf"{sql_start}({sql_keywords}.*?;)", re.IGNORECASE | re.DOTALL
    )
    match = sql_pattern.search(text)
    if match:
        return match.group(1).strip()

    # 3. Inline SQL without semicolon
    fallback_pattern = re.compile(
        rf"{sql_start}({sql_keywords}.*)", re.IGNORECASE | re.DOTALL
    )
    fallback_match = fallback_pattern.search(text)
    if fallback_match:
        return fallback_match.group(1).strip()

    return ""


ALL_DIALECTS = [
    "Athena",
    "BigQuery",
    "ClickHouse",
    "Databricks",
    "Doris",
    "Drill",
    "Druid",
    "DuckDB",
    "Hive",
    "Materialize",
    "MySQL",
    "Oracle",
    "Postgres",
    "Presto",
    "PRQL",
    "Redshift",
    "RisingWave",
    "Snowflake",
    "Spark",
    "Spark2",
    "SQLite",
    "StarRocks",
    "Tableau",
    "Teradata",
    "Trino",
    "TSQL",
]
