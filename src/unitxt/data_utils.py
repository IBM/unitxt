import glob
import json
import os
import sqlite3
from typing import List, Optional

import evaluate
from huggingface_hub import snapshot_download

# Constants for SQL timeout and download lock timeout.
SQL_TIMEOUT = 100
DOWNLOAD_LOCK_TIMEOUT = 1000

# Path to the user's databases cache directory.


# Base headers for HTTP requests.
BASE_HEADERS = {"Content-Type": "application/json", "accept": "application/json"}
# Path to the prompt cache file.


# Logger instance.
logger = evaluate.logging.get_logger(__name__)


# class JSONCache:
#     """A class for managing a JSON cache stored in a file."""

#     def __init__(self, filename):
#         """Initializes the JSON cache."""
#         logger.debug(f"Initializing JSON cache from: {filename}")
#         self.filename = filename
#         self.cache = self.load_cache()
#         self.lock_path = filename + ".lock"
#         self.cache_lock_timeout = 5

#     def load_cache(self):
#         cache = {}
#         if os.path.exists(self.filename):
#             with open(self.filename) as file:
#                 for line in file:
#                     if line.strip():
#                         entry = json.loads(line)
#                         cache.update(entry)
#         return cache

#     def add_to_cache(self, key, value):
#         self.cache[key] = value
#         self.append_to_file({key: value})

#     def append_to_file(self, new_entry):
#         with open(self.filename, "a") as file, FileLock(
#             self.lock_path, timeout=self.cache_lock_timeout
#         ):
#             file.write(json.dumps(new_entry) + "\n")

#     def get_from_cache(self, key):
#         return self.cache.get(key, None)


class SQLData:
    def __init__(self, prompt_cache_location=None):
        # self.base_headers = base_headers if base_headers else BASE_HEADERS

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

    def download_database(self):
        done_file_path = os.path.join(self.databases_folder, "download_done")
        if not os.path.exists(done_file_path):
            snapshot_download(
                repo_id="premai-io/birdbench",
                repo_type="dataset",
                local_dir=self.databases_folder,
                force_download=False,
                allow_patterns="*validation*",
            )
            open(os.path.join(self.databases_folder, "download_done"), "w")

    def get_db_file_path(self, db_name):
        self.download_database()

        db_file_pattern = os.path.join(self.databases_folder, "**", db_name + ".sqlite")
        db_file_paths = glob.glob(db_file_pattern, recursive=True)

        if len(db_file_paths) == 0:
            raise FileNotFoundError(
                f"Database file {db_name} not found. You can try deleting folder {self.databases_folder} and running again."
            )
        if not len(db_file_paths) == 1:
            raise FileExistsError(f"More than one files matched for {db_name}")

        return db_file_paths[0]

    def get_tables_json(self):
        if not self.tables_json:
            self.download_database()
            db_tables_json_file = os.path.join(self.databases_folder, "tables.json")
            if not os.path.exists(db_tables_json_file):
                raise FileNotFoundError(
                    f"tables.json file not found {db_tables_json_file}. "
                    f"You can try deleting folder {self.databases_folder} and running again."
                )
            self.tables_json = json.load(open(db_tables_json_file))
        return self.tables_json

    ## function below from BIRD repo https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/gpt_request.py
    def format_table(self, column_names: list, values: list):
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

    def get_db_schema(
        self,
        db_name: str,
        db_type: str = "sqlite",
        select_tables: Optional[List[str]] = None,
        select_columns: Optional[List[str]] = None,
        num_rows_from_table_to_add: int = 0,
    ) -> str:
        # extract create ddls
        if (
            db_type == "sqlite"
        ):  # this is BIRD style, from https://github.com/AlibabaResearch/DAMO-ConvAI/blob/9406bd7a49e15ff720770263d327fe62d9261325/bird/llm/src/gpt_request.py#L60
            db_path: str = self.get_db_file_path(db_name)
            conn: sqlite3.Connection = sqlite3.connect(db_path)
            # Create a cursor object
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables: list[tuple[str]] = cursor.fetchall()
            schemas: dict[str, str] = {}

            for table in tables:
                if isinstance(table, tuple):
                    table = table[0]
                if table == "sqlite_sequence":
                    continue
                if select_tables:
                    if table.lower() not in select_tables:
                        continue
                sql_query: str = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';"
                cursor.execute(sql_query)
                schema_prompt: str = cursor.fetchone()[0]

                if select_tables and select_columns:
                    schema_prompt = self.apply_column_selection(
                        select_columns, table, schema_prompt
                    )

                if num_rows_from_table_to_add:
                    schema_prompt = self.add_table_rows_to_prompt(
                        num_rows_from_table_to_add, cursor, table, schema_prompt
                    )

                schemas[table] = schema_prompt

            schema_prompt: str = "\n\n".join(list(schemas.values()))
        else:
            raise NotADirectoryError(
                f"only sqlite databases are currently supported, instead {db_type} was selected"
            )
        return schema_prompt

    def add_table_rows_to_prompt(
        self, num_rows_from_table_to_add, cursor, table, schema_prompt
    ):
        cur_table: str = table
        sql_query: str = (
            f"SELECT * FROM `{cur_table}` LIMIT {num_rows_from_table_to_add}"
        )
        cursor.execute(sql_query)
        column_names: list[str] = [description[0] for description in cursor.description]
        values: list[tuple] = cursor.fetchall()
        rows_prompt: str = self.format_table(column_names=column_names, values=values)
        verbose_prompt: str = f"/* \nSample data ({num_rows_from_table_to_add} example row(s)): \n SELECT * FROM {cur_table} LIMIT {num_rows_from_table_to_add}; \n {rows_prompt} \n */"
        return f"{schema_prompt}\n\n{verbose_prompt}"

    def apply_column_selection(self, select_columns, table, schema_prompt):
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

        schema_prompt = "\n".join(lines)
        return schema_prompt  # noqa


if __name__ == "__main__":
    BIRD_DATA = SQLData()
    BIRD_DATA.get_db_schema(
        db_name="formula_1", db_type="sqlite", num_rows_from_table_to_add=1
    )
