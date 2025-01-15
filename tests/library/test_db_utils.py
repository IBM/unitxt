import os
import unittest
from unittest.mock import MagicMock, patch

import requests
from unitxt.db_utils import RemoteDatabaseConnector
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
        )

    @patch("requests.post")
    def test_execute_query_failure(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        connector = RemoteDatabaseConnector(self.db_config)
        result = connector.execute_query("SELECT * FROM table1")

        self.assertIsNone(result)
