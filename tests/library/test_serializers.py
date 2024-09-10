from unitxt.serializers import (
    DefaultSerializer,
    DialogSerializer,
    DynamicSerializer,
    NumberQuantizingSerializer,
    NumberSerializer,
    TableSerializer,
)
from unitxt.types import Dialog, Image, Number, Table, Text, Turn

from tests.utils import UnitxtTestCase


class TestSerializers(UnitxtTestCase):
    def setUp(self):
        self.default_serializer = DefaultSerializer()
        self.dialog_serializer = DialogSerializer()
        self.number_serializer = NumberSerializer()
        self.table_serializer = TableSerializer()
        self.custom_serializer = DynamicSerializer()
        self.custom_serializer_with_number = DynamicSerializer(
            number=NumberSerializer()
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
        image_data = Image(image="fake_image_data")
        instance = {}
        result = self.custom_serializer.serialize(image_data, instance)
        self.assertEqual(
            result, '<img src="media/images/0">'
        )  # Using default serialization
        self.assertEqual(instance, {"media": {"images": ["fake_image_data"]}})

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
