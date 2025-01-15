import os
import tempfile

from unitxt.serializers import (
    DefaultSerializer,
    DialogSerializer,
    MultiTypeSerializer,
    NumberQuantizingSerializer,
    NumberSerializer,
    SQLSchemaSerializer,
    TableSerializer,
)
from unitxt.settings_utils import get_constants
from unitxt.test_utils.operators import check_operator
from unitxt.types import Dialog, Number, SQLSchemaConfig, Table, Text, Turn

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

    def test_local_sqlite_connector(self):
        # Create a temporary SQLite database file for testing
        with tempfile.NamedTemporaryFile(
            suffix=".sqlite", delete=False, mode="w"
        ) as tmpfile:
            db_path = tmpfile.name
        # db_name should be equal to the db_id of one of the instance from one of unitxt datasets.
        # for this test we use the 'concert_singer' dataset.
        db_name = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "databases/concert_singer/concert_singer.sqlite",
        )
        # Copy the content of db_path into the temporary db file
        with open(db_name, "rb") as f:
            with open(db_path, "wb") as f2:
                f2.write(f.read())

        serializer = SQLSchemaSerializer(db_type="sqlite")
        schema_value = SQLSchemaConfig({"db_id": db_name})
        instance = {}

        serialized_schema = serializer.process_instance_value(schema_value, instance)

        # Assertions to validate the serialized schema
        self.assertIn("CREATE TABLE", serialized_schema)
        self.assertIn("singer", serialized_schema)
        self.assertIn("concert", serialized_schema)
        self.assertIn("stadium", serialized_schema)

        # Clean up: delete the temporary file
        os.remove(db_path)

    def test_mock_connector(self):
        mock_db_config = {
            "tables": {
                "singers": {
                    "columns": ["id", "name", "country", "age"],
                    "rows": [[1, "a", "b", 50], [2, "c", "d", 60]],
                },
            }
        }
        serializer = SQLSchemaSerializer(db_type="mock", db_config=mock_db_config)
        schema_value = SQLSchemaConfig({"db_id": "mock_db"})
        instance = {}

        serialized_schema = serializer.process_instance_value(schema_value, instance)

        # Assertions to validate the serialized schema
        self.assertIn("CREATE TABLE", serialized_schema)
        self.assertIn("singers", serialized_schema)

    def test_remote_connector(self):
        # Assuming a way to mock or skip remote calls for testing
        remote_db_config = {
            "api_url": "http://example.com/api",  # Replace with a mock or test URL
            "database_id": "test_db_id",
        }

        serializer = SQLSchemaSerializer(db_type="remote", db_config=remote_db_config)
        schema_value = SQLSchemaConfig({"db_id": "remote_db"})
        instance = {}

        # Mock the requests.post call to avoid actual remote calls
        import requests

        original_post = requests.post

        def mocked_post(*args, **kwargs):
            # Simulate a successful response
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_response.json = lambda: "Mocked Schema Data"
            mock_response.raise_for_status = lambda: None
            return mock_response

        requests.post = mocked_post

        try:
            serialized_schema = serializer.process_instance_value(
                schema_value, instance
            )
        finally:
            requests.post = original_post

        # Assertions to validate the serialized schema or expected behavior
        self.assertEqual(serialized_schema, "Mocked Schema Data")

    def test_invalid_db_type(self):
        serializer = SQLSchemaSerializer(db_type="invalid_type")
        schema_value = SQLSchemaConfig({"db_id": "some_db"})
        instance = {}

        with self.assertRaises(ValueError):
            serializer.process_instance_value(schema_value, instance)

    def test_operator_interface(self):
        operator = SQLSchemaSerializer(db_type="sqlite")
        # Create a dummy database file for testing
        db_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "databases/concert_singer/concert_singer.sqlite",
        )

        inputs = [{"db_id": db_file, "type": "SQLSchema"}]
        targets = [{"db_id": db_file, "type": "SQLSchema", "text": "CREATE TABLE"}]
        check_operator(operator=operator, inputs=inputs, targets=targets)
