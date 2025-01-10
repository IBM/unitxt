import glob
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import evaluate
from huggingface_hub import snapshot_download

# Constants for SQL timeout and download lock timeout.
SQL_TIMEOUT = 100
DOWNLOAD_LOCK_TIMEOUT = 1000

# Path to the user's databases cache directory.
# Logger instance.
logger = evaluate.logging.get_logger(__name__)


class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config

    @abstractmethod
    def get_table_schema(
        self,
        select_tables: Optional[List[str]] = None,
        select_columns: Optional[List[str]] = None,
        num_rows_from_table_to_add: int = 0,
    ) -> str:
        """Abstract method to get database schema."""
        pass

    @abstractmethod
    def format_table(self, column_names: list, values: list) -> str:
        """Abstract method to format table data."""
        pass


class SQLiteConnector(DatabaseConnector):
    """Database connector for SQLite databases."""

    def __init__(self, db_config: Dict[str, Any]):
        super().__init__(db_config)
        self.db_path = self.db_config.get("db_path")
        if not self.db_path:
            raise ValueError("db_path is required for SQLiteConnector.")
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        self.cursor: sqlite3.Cursor = self.conn.cursor()

    def get_table_schema(
        self,
        select_tables: Optional[List[str]] = None,
        select_columns: Optional[List[str]] = None,
        num_rows_from_table_to_add: int = 0,
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
            if select_tables and table.lower() not in select_tables:
                continue
            sql_query: str = (
                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';"
            )
            self.cursor.execute(sql_query)
            schema_prompt: str = self.cursor.fetchone()[0]

            if select_tables and select_columns:
                schema_prompt = self.apply_column_selection(
                    select_columns, table, schema_prompt
                )

            if num_rows_from_table_to_add:
                schema_prompt = self.add_table_rows_to_prompt(
                    num_rows_from_table_to_add, table, schema_prompt
                )
            schemas[table] = schema_prompt

        schema_prompt: str = "\n\n".join(list(schemas.values()))
        return schema_prompt

    def add_table_rows_to_prompt(
        self, num_rows_from_table_to_add: int, table: str, schema_prompt: str
    ) -> str:
        """Adds sample table rows to the schema prompt."""
        cur_table: str = table
        sql_query: str = (
            f"SELECT * FROM `{cur_table}` LIMIT {num_rows_from_table_to_add}"
        )
        self.cursor.execute(sql_query)
        column_names: list[str] = [
            description[0] for description in self.cursor.description
        ]
        values: list[tuple] = self.cursor.fetchall()
        rows_prompt: str = self.format_table(column_names=column_names, values=values)
        verbose_prompt: str = f"/* \nSample data ({num_rows_from_table_to_add} example row(s)): \n SELECT * FROM {cur_table} LIMIT {num_rows_from_table_to_add}; \n {rows_prompt} \n */"
        return f"{schema_prompt}\n\n{verbose_prompt}"

    def apply_column_selection(
        self, select_columns: List[str], table: str, schema_prompt: str
    ) -> str:
        """Filters columns based on `select_columns`."""
        lines: list[str] = []
        for line in schema_prompt.split("\n"):
            if line.startswith("    "):
                col_name: str = line.strip().split()[0]
                if "`" in line:
                    col_name = line[line.find("`") + 1 : line.rfind("`")]
                if col_name.endswith(","):
                    lines.append(line)
                    continue
                col_name_formatted: str = table + "." + col_name.lower()
                if col_name_formatted in select_columns:
                    lines.append(line)
            else:
                lines.append(line)

        return "\n".join(lines)

    def format_table(self, column_names: list, values: list) -> str:
        """Formats table data into a string for display."""
        rows = []
        # Determine the maximum width of each column
        widths = [
            max(len(str(value[i])) for value in [*values, column_names])
            for i in range(len(column_names))
        ]
        header = "".join(
            f"{column.rjust(width)} " for column, width in zip(column_names, widths)
        )
        for value in values:
            row = "".join(f"{str(v).rjust(width)} " for v, width in zip(value, widths))
            rows.append(row)
        rows = "\n".join(rows)
        return header + "\n" + rows


class MockConnector(DatabaseConnector):
    """Database connector for mocking databases with in-memory data structures."""

    def __init__(self, db_config: Dict[str, Any]):
        super().__init__(db_config)
        self.tables = db_config.get("tables", {})
        if not self.tables:
            raise ValueError("tables is required for MockConnector.")

    def get_table_schema(
        self,
        select_tables: Optional[List[str]] = None,
        select_columns: Optional[List[str]] = None,
        num_rows_from_table_to_add: int = 0,
    ) -> str:
        """Generates a mock schema from the tables structure."""
        schemas = {}
        for table_name, table_data in self.tables.items():
            if select_tables and table_name.lower() not in select_tables:
                continue
            columns = ", ".join([f"`{col}` TEXT" for col in table_data["columns"]])
            schema = f"CREATE TABLE `{table_name}` ({columns});"
            if num_rows_from_table_to_add:
                schema = self.add_table_rows_to_prompt(
                    num_rows_from_table_to_add, table_name, schema
                )

            schemas[table_name] = schema

        return "\n\n".join(list(schemas.values()))

    def add_table_rows_to_prompt(
        self, num_rows_from_table_to_add: int, table_name: str, schema_prompt: str
    ) -> str:
        """Adds mock table rows to the schema prompt."""
        table_data = self.tables.get(table_name)
        if not table_data:
            return schema_prompt  # Return original schema if table not found

        rows = table_data.get("rows", [])[:num_rows_from_table_to_add]
        if not rows:
            return schema_prompt

        rows_prompt: str = self.format_table(
            column_names=table_data["columns"], values=rows
        )
        verbose_prompt: str = f"/* \nSample data ({num_rows_from_table_to_add} example row(s)): \n SELECT * FROM {table_name} LIMIT {num_rows_from_table_to_add}; \n {rows_prompt} \n */"
        return f"{schema_prompt}\n\n{verbose_prompt}"

    def format_table(self, column_names: list, values: list) -> str:
        """Formats table data into a string for display."""
        rows = []
        # Determine the maximum width of each column
        widths = [
            max(len(str(value[i])) for value in [*values, column_names])
            for i in range(len(column_names))
        ]
        header = "".join(
            f"{column.rjust(width)} " for column, width in zip(column_names, widths)
        )
        for value in values:
            row = "".join(f"{str(v).rjust(width)} " for v, width in zip(value, widths))
            rows.append(row)
        rows = "\n".join(rows)
        return header + "\n" + rows


class SQLData:
    """Manages SQL database connections and schemas."""

    def __init__(self, prompt_cache_location=None):
        self.prompt_cache_location = os.path.join(
            os.environ.get("UNITXT_TEXT2SQL_CACHE", "cache/text2sql"),
            "user_cache",
            "text2sql_requests_cache.jsonl",
        )

        self.databases_folder = os.path.join(
            os.environ.get("UNITXT_TEXT2SQL_CACHE", "cache/text2sql"), "databases"
        )
        os.makedirs(self.databases_folder, exist_ok=True)
        self.tables_json = None
        # self.prompt_cache = JSONCache(self.prompt_cache_location)

    def download_database(self, db_name):
        """Downloads the dataset from huggingface if needed."""
        done_file_path = os.path.join(self.databases_folder, "download_done")
        if "bird/" in db_name:
            if not os.path.exists(done_file_path):
                snapshot_download(
                    repo_id="premai-io/birdbench",
                    repo_type="dataset",
                    local_dir=self.databases_folder,
                    force_download=False,
                    allow_patterns="*validation*",
                )
                open(os.path.join(self.databases_folder, "download_done"), "w").close()

    def get_db_file_path(self, db_name):
        """Gets the local path of a downloaded database file."""
        self.download_database(db_name)
        db_name = db_name.split("/")[-1]

        db_file_pattern = os.path.join(self.databases_folder, "**", db_name + ".sqlite")
        db_file_paths = glob.glob(db_file_pattern, recursive=True)

        if not db_file_paths:
            raise FileNotFoundError(f"Database file {db_name} not found.")
        if len(db_file_paths) > 1:
            raise FileExistsError(f"More than one files matched for {db_name}")
        return db_file_paths[0]

    # def get_tables_json(self):
    #     """Gets the tables.json file"""
    #     if not self.tables_json:
    #         self.download_database()
    #         db_tables_json_file = os.path.join(self.databases_folder, "tables.json")
    #         if not os.path.exists(db_tables_json_file):
    #             raise FileNotFoundError(
    #                 f"tables.json file not found {db_tables_json_file}. "
    #                 f"You can try deleting folder {self.databases_folder} and running again."
    #             )
    #         self.tables_json = json.load(open(db_tables_json_file))
    #     return self.tables_json

    def get_db_schema(
        self,
        db_name: str,
        db_type: str = "local",
        select_tables: Optional[List[str]] = None,
        select_columns: Optional[List[str]] = None,
        num_rows_from_table_to_add: int = 0,
        mock_db: Optional[Dict] = None,
    ) -> str:
        """Retrieves the schema of a database."""
        if db_type == "local":
            db_path = self.get_db_file_path(db_name)
            db_config = {"db_path": db_path}
            connector = SQLiteConnector(db_config)
        elif db_type == "mock":
            if not mock_db:
                raise ValueError("db_type is mock, but mock_db was not given")
            connector = MockConnector(db_config={"tables": mock_db})
        else:
            raise ValueError(
                f"Unsupported database type: {db_type}. Use 'sqlite' or 'mock'."
            )

        return connector.get_table_schema(
            select_tables, select_columns, num_rows_from_table_to_add
        )


if __name__ == "__main__":
    # Example for using a downloaded sqlite db
    sql_data_manager = SQLData()
    schema_from_sqlite = sql_data_manager.get_db_schema(
        db_name="bird/california_schools",
        db_type="local",
    )

    # Example for using a mock db
    mock_db_data = {
        "users": {
            "columns": ["id", "name", "age"],
            "rows": [
                [1, "Alice", 30],
                [2, "Bob", 25],
            ],
        },
        "products": {
            "columns": ["id", "name", "price"],
            "rows": [
                [1, "Laptop", 1200],
                [2, "Monitor", 300],
            ],
        },
    }
    schema_from_mock = sql_data_manager.get_db_schema(
        db_name="mock_db_name",
        db_type="mock",
        select_tables=["users", "products"],
        mock_db=mock_db_data,
    )
