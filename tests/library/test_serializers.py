import unittest
from unittest.mock import MagicMock, patch

from unitxt.serializers import (
    DefaultSerializer,
    DialogSerializer,
    MultiTypeSerializer,
    NumberQuantizingSerializer,
    NumberSerializer,
    SQLDatabaseAsSchemaSerializer,
    TableSerializer,
)
from unitxt.settings_utils import get_constants
from unitxt.types import Dialog, Number, SQLDatabase, Table, Text, Turn

from tests.library.test_image_operators import create_random_jpeg_image
from tests.utils import UnitxtTestCase

constants = get_constants()


class TestSerializers(UnitxtTestCase):
    def setUp(self):
        self.default_serializer = DefaultSerializer()
        self.dialog_serializer = DialogSerializer()
        self.number_serializer = NumberSerializer()
        self.table_serializer = TableSerializer()
        self.custom_serializer = MultiTypeSerializer()
        self.custom_serializer_with_number = MultiTypeSerializer(
            serializers=[NumberSerializer()]
        )
        self.number_quantizing_serializer = NumberQuantizingSerializer(quantum=0.2)

    def test_default_serializer_with_string(self):
        result = self.default_serializer.serialize("test", {})
        self.assertEqual(result, "test")

    def test_default_serializer_with_number_no_serialization(self):
        result = self.default_serializer.serialize(123, {})
        self.assertEqual(result, "123")

    def test_default_serializer_with_number(self):
        result = self.custom_serializer_with_number.serialize(123, {})
        self.assertEqual(result, "123")

    def test_default_serializer_with_dict(self):
        test_dict = {"key": "value"}
        result = self.default_serializer.serialize(test_dict, {})
        self.assertEqual(result, "{'key': 'value'}")

    def test_dialog_serializer(self):
        dialog_data = Dialog(
            [Turn(role="user", content="Hello"), Turn(role="agent", content="Hi there")]
        )
        expected_output = "user: Hello\nagent: Hi there"
        result = self.dialog_serializer.serialize(dialog_data, {})
        self.assertEqual(result, expected_output)

    def test_number_serializer_with_integer(self):
        number_data = Number(42)
        result = self.number_serializer.serialize(number_data, {})
        self.assertEqual(result, "42")

    def test_number_serializer_with_float(self):
        number_data = Number(42.123)
        result = self.number_serializer.serialize(number_data, {})
        self.assertEqual(result, "42.1")

    def test_number_quantizing_serializer_with_int_quantum(self):
        serializer = NumberQuantizingSerializer(quantum=2)
        result = serializer.serialize(31, {})
        self.assertEqual(result, "32")
        serializer = NumberQuantizingSerializer(quantum=1)
        result = serializer.serialize(31, {})
        self.assertEqual(result, "31")
        serializer = NumberQuantizingSerializer(quantum=2)
        result = serializer.serialize(31.1, {})
        self.assertEqual(result, "32")
        serializer = NumberQuantizingSerializer(quantum=1)
        result = serializer.serialize(31.1, {})
        self.assertEqual(result, "31")

    def test_number_quantizing_serializer_with_float_quantum(self):
        serializer = NumberQuantizingSerializer(quantum=0.2)
        result = serializer.serialize(31, {})
        self.assertEqual(result, "31.0")
        serializer = NumberQuantizingSerializer(quantum=0.2)
        result = serializer.serialize(31.5, {})
        self.assertEqual(result, "31.6")
        serializer = NumberQuantizingSerializer(quantum=0.2)
        result = serializer.serialize(31.1, {})
        self.assertEqual(result, "31.2")
        serializer = NumberQuantizingSerializer(quantum=0.2)
        result = serializer.serialize(29.999, {})
        self.assertEqual(result, "30.0")

    def test_table_serializer(self):
        table_data = Table(header=["col1", "col2"], rows=[[1, 2], [3, 4]])
        expected_output = "col1,col2\n1,2\n3,4"
        result = self.table_serializer.serialize(table_data, {})
        self.assertEqual(result, expected_output)

    def test_custom_serializer_with_image(self):
        image = create_random_jpeg_image(10, 10, 1)
        image_data = {"image": image, "format": "JPEG"}
        instance = {}
        result = self.custom_serializer.serialize(image_data, instance)
        self.assertEqual(
            result, f'<{constants.image_tag} src="media/images/0">'
        )  # Using default serialization
        # self.assertEqual(instance, {"media": {"images": [image]}})

    def test_custom_serializer_with_table(self):
        table_data = Table(header=["col1", "col2"], rows=[[1, 2], [3, 4]])
        expected_output = "col1,col2\n1,2\n3,4"
        result = self.custom_serializer.serialize(table_data, {})
        self.assertEqual(result, expected_output)

    def test_custom_serializer_with_dialog(self):
        dialog_data = Dialog(
            [Turn(role="user", content="Hello"), Turn(role="agent", content="Hi there")]
        )
        result = self.custom_serializer.serialize(dialog_data, {})
        self.assertEqual(
            result, "user: Hello\nagent: Hi there"
        )  # Using default serialization

    def test_custom_serializer_with_text(self):
        text_data = Text("Sample text")
        result = self.custom_serializer.serialize(text_data, {})
        self.assertEqual(
            result, "Sample text"
        )  # Since Text is a NewType of str, it should return the string directly

    def test_custom_serializer_with_unrecognized_type(self):
        custom_object = {"key": "value"}
        result = self.custom_serializer.serialize(custom_object, {})
        self.assertEqual(
            result, "{'key': 'value'}"
        )  # Should fall back to str conversion

    def test_custom_serializer_with_number(self):
        number_data = Number(42)
        result = self.custom_serializer.serialize(number_data, {})
        self.assertEqual(result, "42")  # Should return the number as a string


class TestSQLDatabaseAsSchemaSerializer(unittest.TestCase):
    def test_serialize_in_memory_success(self):
        db_config: SQLDatabase = {
            "db_type": "in_memory",
            "db_id": None,
            "dbms": None,
            "data": {
                "table1": {"columns": ["col1", "col2"], "rows": [[1, "a"], [2, "b"]]},
                "table2": {"columns": ["name", "age"], "rows": [["Alice", 30]]},
            },
        }

        serializer = SQLDatabaseAsSchemaSerializer()
        result = serializer.serialize(db_config, {})
        expected_schema = (
            "CREATE TABLE `table1` (`col1` TEXT, `col2` TEXT);\n\n"
            "CREATE TABLE `table2` (`name` TEXT, `age` TEXT);"
        )
        self.assertEqual(result, expected_schema)

    @patch.dict(
        "os.environ",
        {"SQL_API_KEY": "test_api_key"},  # pragma: allowlist secret
        clear=True,
    )  # pragma: allowlist-secret
    @patch("requests.post")
    def test_serialize_remote_success(self, mock_post):
        db_config: SQLDatabase = {
            "db_type": "remote",
            "db_id": "https://testapi.com/api,db_id=test_db_id",
            "dbms": None,
            "data": None,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "schema": {
                "tables": [
                    {"table_name": "table1", "columns": [{"column_name": "col1"}]},
                    {"table_name": "table2", "columns": [{"column_name": "name"}]},
                ]
            }
        }
        mock_post.return_value = mock_response

        serializer = SQLDatabaseAsSchemaSerializer()
        result = serializer.serialize(db_config, {})

        expected_schema = (
            "Table: table1 has columns: ['col1']\n"
            "Table: table2 has columns: ['name']\n"
        )
        self.assertEqual(result, expected_schema)

    def test_serialize_unsupported_db_type(self):
        db_config: SQLDatabase = {
            "db_type": "unsupported",
            "db_id": "test_db_id",
            "dbms": None,
            "data": None,
        }

        serializer = SQLDatabaseAsSchemaSerializer()
        with self.assertRaises(ValueError):
            serializer.serialize(db_config, {})
