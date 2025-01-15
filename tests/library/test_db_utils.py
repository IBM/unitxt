import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests
from unitxt.db_utils import (
    InMemoryDatabaseConnector,
    LocalSQLiteConnector,
    RemoteDatabaseConnector,
)
from unitxt.types import SQLDatabase


class TestRemoteDatabaseConnector(unittest.TestCase):
    def setUp(self):
        # Set up any necessary environment variables or configurations
        self.env_patcher = patch.dict(
            os.environ,
            {"SQL_API_KEY": "test_api_key"},  # pragma: allowlist-secret
            clear=True,
        )
        self.env_patcher.start()

        self.db_config: SQLDatabase = {
            "db_type": "remote",
            "db_id": "https://testapi.com/api,db_id=test_db_id",
            "dbms": None,
            "data": None,
        }

    def tearDown(self):
        # Clean up any resources or configurations
        self.env_patcher.stop()

    def test_init_success(self):
        connector = RemoteDatabaseConnector(self.db_config)
        self.assertEqual(connector.api_url, "https://testapi.com/api")
        self.assertEqual(connector.database_id, "test_db_id")
        self.assertEqual(connector.api_key, "test_api_key")
        self.assertEqual(connector.base_headers["Authorization"], "Bearer test_api_key")

    def test_init_missing_api_url(self):
        self.db_config["db_id"] = ",db_id=test_db_id"
        with self.assertRaises(ValueError):
            RemoteDatabaseConnector(self.db_config)

    def test_init_missing_api_key(self):
        os.environ.pop("SQL_API_KEY")
        with self.assertRaises(ValueError):
            RemoteDatabaseConnector(self.db_config)

    @patch("requests.post")
    def test_get_table_schema_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "schema": {
                "tables": [
                    {"table_name": "table1", "columns": [{"column_name": "tab1col1"}]},
                    {"table_name": "table2", "columns": [{"column_name": "tab2col1"}]},
                ]
            }
        }
        mock_post.return_value = mock_response

        connector = RemoteDatabaseConnector(self.db_config)
        schema_text = connector.get_table_schema()

        self.assertEqual(
            schema_text,
            "Table: table1 has columns: ['tab1col1']\nTable: table2 has columns: ['tab2col1']\n",
        )

    @patch("requests.post")
    def test_get_table_schema_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        connector = RemoteDatabaseConnector(self.db_config)

        with self.assertRaises(OSError):
            connector.get_table_schema()

    @patch("requests.post")
    def test_execute_query_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        connector = RemoteDatabaseConnector(self.db_config)
        result = connector.execute_query("SELECT * FROM table1")

        self.assertEqual(result, {"result": "success"})
        mock_post.assert_called_once_with(
            "https://testapi.com/api/sql",
            headers=connector.base_headers,
            json={"sql": "SELECT * FROM table1", "dataSourceId": "test_db_id"},
            verify=True,
            timeout=RemoteDatabaseConnector.TIMEOUT,
        )

    @patch("requests.post")
    def test_execute_query_failure(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        connector = RemoteDatabaseConnector(self.db_config)
        result = connector.execute_query("SELECT * FROM table1")

        self.assertIsNone(result)


class TestLocalSQLiteConnector(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a dummy SQLite database
        self.db_id = "test_db"
        self.db_path = os.path.join(self.temp_dir.name, self.db_id + ".sqlite")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE table1 (tab1col1 TEXT, tab1col2 INTEGER)")
        cursor.execute(
            "INSERT INTO table1 VALUES ('value1', 1), ('value2', 2)"
        )  # Insert data into table1
        cursor.execute("CREATE TABLE table2 (tab2col1 REAL, tab2col2 TEXT)")
        cursor.execute(
            "INSERT INTO table2 VALUES (3.14, 'pi'), (2.71, 'e')"
        )  # Insert data into table2
        cursor.execute("CREATE TABLE sequence (name,seq)")
        cursor.execute("INSERT INTO sequence VALUES ('table1', 2), ('table2', 2)")
        conn.commit()
        conn.close()

        self.db_config: SQLDatabase = {
            "db_type": "local",
            "db_id": self.db_id,
            "dbms": "sqlite",
            "data": None,
        }

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    @patch(
        "unitxt.db_utils.LocalSQLiteConnector.get_db_file_path",
        side_effect=FileNotFoundError("Database file not found."),
    )
    def test_init_database_not_found(self, mock_get_db_file_path):
        with self.assertRaises(FileNotFoundError):
            LocalSQLiteConnector(self.db_config)

    @patch(
        "unitxt.db_utils.LocalSQLiteConnector.get_db_file_path",
        side_effect=FileExistsError("More than one file matched for db_id"),
    )
    def test_init_multiple_databases_found(self, mock_get_db_file_path):
        with self.assertRaises(FileExistsError):
            LocalSQLiteConnector(self.db_config)


class TestInMemoryDatabaseConnector(unittest.TestCase):
    def setUp(self):
        self.db_config: SQLDatabase = {
            "db_type": "in_memory",
            "db_id": None,
            "dbms": None,
            "data": {
                "users": {
                    "columns": ["user_id", "name", "email", "age", "city"],
                    "rows": [
                        [1, "Alice", "alice@example.com", 30, "New York"],
                        [2, "Bob", "bob@example.com", 25, "Los Angeles"],
                        [3, "Charlie", "charlie@example.com", 40, "Chicago"],
                        [4, "David", "david@example.com", 35, "New York"],
                        [5, "Eva", "eva@example.com", 28, "Los Angeles"],
                    ],
                },
                "orders": {
                    "columns": ["order_id", "user_id", "product", "quantity", "price"],
                    "rows": [
                        [101, 1, "Laptop", 2, 1200.00],
                        [102, 1, "Mouse", 5, 25.50],
                        [103, 2, "Keyboard", 3, 75.00],
                        [104, 3, "Monitor", 1, 300.00],
                        [105, 3, "USB Drive", 10, 15.00],
                        [106, 4, "Headphones", 2, 100.00],
                        [107, 5, "Webcam", 1, 80.00],
                        [108, 5, "Printer", 1, 250.00],
                        [109, 5, "Laptop", 1, 1300.00],
                        [110, 5, "Mouse", 2, 24.00],
                    ],
                },
            },
        }

    def test_init_success(self):
        connector = InMemoryDatabaseConnector(self.db_config)
        self.assertEqual(connector.tables, self.db_config["data"])

    def test_init_missing_tables(self):
        self.db_config["data"] = None
        with self.assertRaises(ValueError):
            InMemoryDatabaseConnector(self.db_config)

    def test_get_table_schema(self):
        connector = InMemoryDatabaseConnector(self.db_config)
        schema_text = connector.get_table_schema()
        expected_schema = (
            "CREATE TABLE `users` (`user_id` TEXT, `name` TEXT, `email` TEXT, `age` TEXT, `city` TEXT);\n\n"
            "CREATE TABLE `orders` (`order_id` TEXT, `user_id` TEXT, `product` TEXT, `quantity` TEXT, `price` TEXT);"
        )
        self.assertEqual(schema_text, expected_schema)

    def test_get_table_schema_with_selected_tables(self):
        connector = InMemoryDatabaseConnector(self.db_config)
        schema_text = connector.get_table_schema(select_tables=["orders"])
        expected_schema = "CREATE TABLE `orders` (`order_id` TEXT, `user_id` TEXT, `product` TEXT, `quantity` TEXT, `price` TEXT);"

        self.assertEqual(schema_text, expected_schema)

    def test_execute_query_success(self):
        connector = InMemoryDatabaseConnector(self.db_config)
        result = connector.execute_query("SELECT * FROM users WHERE age > 30")
        expected_result = [
            (3, "Charlie", "charlie@example.com", 40, "Chicago"),
            (4, "David", "david@example.com", 35, "New York"),
        ]

        self.assertEqual(result, expected_result)

    def test_execute_query_failure(self):
        connector = InMemoryDatabaseConnector(self.db_config)
        result = connector.execute_query("SELECT * FROM non_existent_table")

        self.assertIsNone(result)

    def test_execute_complex_query(self):
        connector = InMemoryDatabaseConnector(self.db_config)
        query = """
            SELECT u.name, o.product, o.quantity
            FROM users u
            JOIN orders o ON u.user_id = o.user_id
            WHERE u.city = 'Los Angeles'
            ORDER BY o.quantity DESC
        """
        result = connector.execute_query(query)
        expected_result = [
            ("Bob", "Keyboard", 3),
            ("Eva", "Mouse", 2),
            ("Eva", "Laptop", 1),
            ("Eva", "Printer", 1),
            ("Eva", "Webcam", 1),
        ]
        self.assertEqual(result, expected_result)

    def test_execute_query_with_aggregation(self):
        connector = InMemoryDatabaseConnector(self.db_config)
        query = """
            SELECT u.city, AVG(u.age)
            FROM users u
            GROUP BY u.city
            ORDER BY u.city ASC
        """
        result = connector.execute_query(query)
        expected_result = [
            ("Chicago", 40.0),
            ("Los Angeles", 26.5),
            ("New York", 32.5),
        ]
        self.assertEqual(result, expected_result)

    def test_execute_query_with_sum_and_having(self):
        connector = InMemoryDatabaseConnector(self.db_config)
        query = """
            SELECT u.name, SUM(o.price)
            FROM users u
            JOIN orders o on u.user_id = o.user_id
            GROUP BY u.name
            HAVING SUM(o.price) > 300
            ORDER BY u.name DESC
        """
        result = connector.execute_query(query)
        expected_result = [("Eva", 1654.0), ("Charlie", 315.0), ("Alice", 1225.5)]

        self.assertEqual(result, expected_result)
