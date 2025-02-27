import glob
import os
import re
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
    """Return True if both SQL queries are naively considered equivalent."""
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
