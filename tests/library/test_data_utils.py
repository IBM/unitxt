import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from unitxt.db_utils import (  # Replace 'your_module' with the actual module name
    LocalSQLiteConnector,
    MockConnector,
    SQLData,
)


class TestSQLiteConnector(unittest.TestCase):
    def setUp(self):
        # Create a temporary SQLite database for testing
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        self.db_path = self.temp_db_file.name
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create a sample table
        self.cursor.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER
            )
            """
        )
        self.cursor.execute(
            "INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25)"
        )
        self.conn.commit()

        self.db_config = {"db_path": self.db_path}
        self.connector = LocalSQLiteConnector(self.db_config)

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_init_missing_db_path(self):
        with self.assertRaisesRegex(ValueError, "db_path is required"):
            LocalSQLiteConnector({})

    def test_get_table_schema(self):
        expected_schema = (
            "CREATE TABLE users (\n"
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "    name TEXT NOT NULL,\n"
            "    age INTEGER\n"
            ")"
        )
        schema = self.connector.get_table_schema()
        self.assertIn(expected_schema, schema)

    def test_get_table_schema_with_select_tables(self):
        schema = self.connector.get_table_schema(select_tables=["users"])
        self.assertIn("CREATE TABLE users", schema)

        schema = self.connector.get_table_schema(select_tables=["nonexistent_table"])
        self.assertEqual(schema, "")

    def test_get_table_schema_with_select_columns(self):
        expected_schema = (
            "CREATE TABLE users (\n" "    name TEXT NOT NULL\n" ")"
        )  # Only 'name' column
        schema = self.connector.get_table_schema(
            select_tables=["users"], select_columns=["users.name"]
        )
        self.assertIn(expected_schema, schema)

    def test_get_table_schema_with_num_rows(self):
        schema = self.connector.get_table_schema(
            select_tables=["users"], num_rows_from_table_to_add=1
        )
        self.assertIn("/* \nSample data (1 example row(s))", schema)
        self.assertIn("Alice", schema)

    def test_add_table_rows_to_prompt(self):
        schema_prompt = "CREATE TABLE test (id INTEGER, name TEXT);"
        expected_prompt = (
            "CREATE TABLE test (id INTEGER, name TEXT);\n\n"
            "/* \nSample data (1 example row(s)): \n"
            " SELECT * FROM `users` LIMIT 1; \n"
            " id name age \n"
            "  1 Alice 30 \n"
            " */"
        )
        new_prompt = self.connector.add_table_rows_to_prompt(1, "users", schema_prompt)
        self.assertEqual(new_prompt, expected_prompt)

    def test_apply_column_selection(self):
        schema_prompt = (
            "CREATE TABLE users (\n"
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "    name TEXT NOT NULL,\n"
            "    age INTEGER\n"
            ")"
        )
        expected_prompt = "CREATE TABLE users (\n" "    name TEXT NOT NULL\n" ")"
        new_prompt = self.connector.apply_column_selection(
            ["users.name"], "users", schema_prompt
        )
        self.assertEqual(new_prompt, expected_prompt)

    def test_format_table(self):
        column_names = ["id", "name", "age"]
        values = [(1, "Alice", 30), (2, "Bob", 25)]
        expected_output = "id name age \n 1 Alice 30 \n 2   Bob 25 "
        formatted_table = self.connector.format_table(column_names, values)
        self.assertEqual(formatted_table, expected_output)


class TestMockConnector(unittest.TestCase):
    def setUp(self):
        self.db_config = {
            "tables": {
                "users": {
                    "columns": ["id", "name", "age"],
                    "rows": [(1, "Alice", 30), (2, "Bob", 25)],
                }
            }
        }
        self.connector = MockConnector(self.db_config)

    def test_init_missing_tables(self):
        with self.assertRaisesRegex(ValueError, "tables is required"):
            MockConnector({})

    def test_get_table_schema(self):
        expected_schema = "CREATE TABLE `users` (`id` TEXT, `name` TEXT, `age` TEXT);"
        schema = self.connector.get_table_schema()
        self.assertEqual(schema, expected_schema)

    def test_get_table_schema_with_select_tables(self):
        schema = self.connector.get_table_schema(select_tables=["users"])
        self.assertIn("CREATE TABLE `users`", schema)

        schema = self.connector.get_table_schema(select_tables=["nonexistent_table"])
        self.assertEqual(schema, "")

    def test_get_table_schema_with_num_rows(self):
        schema = self.connector.get_table_schema(
            select_tables=["users"], num_rows_from_table_to_add=1
        )
        self.assertIn("/* \nSample data (1 example row(s))", schema)
        self.assertIn("Alice", schema)

    def test_add_table_rows_to_prompt(self):
        schema_prompt = "CREATE TABLE `test` (`id` TEXT, `name` TEXT);"
        expected_prompt = (
            "CREATE TABLE `test` (`id` TEXT, `name` TEXT);\n\n"
            "/* \nSample data (1 example row(s)): \n"
            " SELECT * FROM users LIMIT 1; \n"
            "   id  name age \n"
            "    1 Alice 30 \n"
            " */"
        )

        new_prompt = self.connector.add_table_rows_to_prompt(1, "users", schema_prompt)
        self.assertEqual(new_prompt, expected_prompt)

        # Test with a table that doesn't exist
        new_prompt = self.connector.add_table_rows_to_prompt(
            1, "nonexistent_table", schema_prompt
        )
        self.assertEqual(new_prompt, schema_prompt)  # Should return the original schema

        # Test with num_rows_from_table_to_add larger than available rows
        new_prompt = self.connector.add_table_rows_to_prompt(5, "users", schema_prompt)
        self.assertIn("Alice", new_prompt)
        self.assertIn("Bob", new_prompt)
        self.assertNotIn("Charlie", new_prompt)

    def test_format_table(self):
        column_names = ["id", "name", "age"]
        values = [(1, "Alice", 30), (2, "Bob", 25)]
        expected_output = "id name age \n 1 Alice 30 \n 2   Bob 25 "
        formatted_table = self.connector.format_table(column_names, values)
        self.assertEqual(formatted_table, expected_output)


class TestSQLData(unittest.TestCase):
    @patch.dict(os.environ, {"UNITXT_TEXT2SQL_CACHE": ""})
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sql_data = SQLData(prompt_cache_location=self.temp_dir.name)
        self.sql_data.databases_folder = os.path.join(self.temp_dir.name, "databases")
        os.makedirs(self.sql_data.databases_folder, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("your_module.snapshot_download")  # Replace your_module
    def test_download_database(self, mock_snapshot_download):
        db_name = "bird/dev"
        self.sql_data.download_database(db_name)
        mock_snapshot_download.assert_called_once_with(
            repo_id="premai-io/birdbench",
            repo_type="dataset",
            local_dir=self.sql_data.databases_folder,
            force_download=False,
            allow_patterns="*validation*",
        )
        # check that done file is created
        self.assertTrue(
            os.path.exists(
                os.path.join(self.sql_data.databases_folder, "download_done")
            )
        )

        # Test when done file exists - snapshot_download shouldn't be called again
        mock_snapshot_download.reset_mock()
        self.sql_data.download_database(db_name)
        mock_snapshot_download.assert_not_called()

    @patch("your_module.snapshot_download")  # Replace your_module
    def test_download_database_custom_cache(self, mock_snapshot_download):
        custom_cache_dir = tempfile.TemporaryDirectory()
        with patch.dict(os.environ, {"UNITXT_TEXT2SQL_CACHE": custom_cache_dir.name}):
            sql_data = SQLData(prompt_cache_location="dummy")
            sql_data.databases_folder = os.path.join(custom_cache_dir.name, "databases")

            db_name = "bird/dev"
            sql_data.download_database(db_name)

            mock_snapshot_download.assert_called_once_with(
                repo_id="premai-io/birdbench",
                repo_type="dataset",
                local_dir=sql_data.databases_folder,
                force_download=False,
                allow_patterns="*validation*",
            )
        custom_cache_dir.cleanup()

    def test_get_db_file_path(self):
        # Create a dummy database file
        db_name = "test_db"
        db_file_path = os.path.join(self.sql_data.databases_folder, db_name + ".sqlite")
        open(db_file_path, "w").close()

        retrieved_path = self.sql_data.get_db_file_path(db_name)
        self.assertEqual(retrieved_path, db_file_path)

    def test_get_db_file_path_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.sql_data.get_db_file_path("nonexistent_db")

    def test_get_db_file_path_multiple_matches(self):
        # Create multiple dummy database files
        db_name = "test_db"
        db_file_path1 = os.path.join(
            self.sql_data.databases_folder, db_name + ".sqlite"
        )
        db_file_path2 = os.path.join(
            self.sql_data.databases_folder, "subdir", db_name + ".sqlite"
        )
        os.makedirs(os.path.join(self.sql_data.databases_folder, "subdir"))
        open(db_file_path1, "w").close()
        open(db_file_path2, "w").close()

        with self.assertRaises(FileExistsError):
            self.sql_data.get_db_file_path(db_name)
