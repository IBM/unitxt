
import unittest
from typing import Any, Dict, List, Union

from unitxt.tool_calling import (
    convert_chat_api_format_to_tool,
    convert_to_chat_api_format,
    json_schema_to_python_type,
)


class TestConvertToChatAPIFormat(unittest.TestCase):
    def test_basic_conversion(self):
        """Test basic conversion of a simple tool to Chat API format."""
        tool = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": [
                {"name": "param1", "type": str},
                {"name": "param2", "type": int}
            ]
        }

        result = convert_to_chat_api_format(tool)

        # Check main structure
        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "test_tool")
        self.assertEqual(result["function"]["description"], "A test tool")

        # Check parameter schema
        schema = result["function"]["parameters"]
        self.assertEqual(schema["title"], "test_toolParams")
        self.assertEqual(schema["type"], "object")
        self.assertEqual(set(schema["required"]), {"param1", "param2"})
        self.assertEqual(schema["properties"]["param1"]["type"], "string")
        self.assertEqual(schema["properties"]["param2"]["type"], "integer")

    def test_complex_parameters(self):
        """Test conversion with complex parameter types."""
        tool = {
            "name": "complex_tool",
            "description": "A complex tool",
            "parameters": [
                {"name": "list_param", "type": List[str]},
                {"name": "dict_param", "type": Dict[str, Any]},
                {"name": "any_param", "type": Any}
            ]
        }

        result = convert_to_chat_api_format(tool)

        # Check parameter schema
        schema = result["function"]["parameters"]
        self.assertEqual(schema["properties"]["list_param"]["type"], "array")
        self.assertEqual(schema["properties"]["list_param"]["items"]["type"], "string")
        self.assertEqual(schema["properties"]["dict_param"]["type"], "object")
        # For Any type, the schema should be empty or very minimal
        self.assertIn("any_param", schema["properties"])


class TestConvertChatAPIFormatToTool(unittest.TestCase):
    def test_basic_conversion(self):
        """Test basic conversion from Chat API format back to Tool."""
        chat_api_tool = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "integer"}
                    }
                }
            }
        }

        expected_tool = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": [
                {"name": "param1", "type": str},
                {"name": "param2", "type": int}
            ]
        }

        result = convert_chat_api_format_to_tool(chat_api_tool)

        self.assertEqual(result["name"], expected_tool["name"])
        self.assertEqual(result["description"], expected_tool["description"])

        # Check that parameters are correctly converted
        params_by_name = {p["name"]: p for p in result["parameters"]}
        self.assertEqual(len(params_by_name), 2)
        self.assertEqual(params_by_name["param1"]["type"], str)
        self.assertEqual(params_by_name["param2"]["type"], int)

    def test_alternate_format(self):
        """Test conversion with an alternate format (without nested 'function')."""
        chat_api_tool = {
            "name": "alt_tool",
            "description": "An alternate tool",
            "parameters": {
                "properties": {
                    "param1": {"type": "string"}
                }
            }
        }

        result = convert_chat_api_format_to_tool(chat_api_tool)

        self.assertEqual(result["name"], "alt_tool")
        self.assertEqual(result["description"], "An alternate tool")
        self.assertEqual(len(result["parameters"]), 1)
        self.assertEqual(result["parameters"][0]["name"], "param1")
        self.assertEqual(result["parameters"][0]["type"], str)

    def test_complex_parameter_types(self):
        """Test conversion with complex parameter types."""
        chat_api_tool = {
            "type": "function",
            "function": {
                "name": "complex_tool",
                "description": "A complex tool",
                "parameters": {
                    "properties": {
                        "array_param": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "object_param": {"type": "object"},
                        "union_param": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"}
                            ]
                        }
                    }
                }
            }
        }

        result = convert_chat_api_format_to_tool(chat_api_tool)

        params_by_name = {p["name"]: p for p in result["parameters"]}

        # Check array parameter
        self.assertEqual(params_by_name["array_param"]["type"].__origin__, list)
        self.assertEqual(params_by_name["array_param"]["type"].__args__[0], str)

        # Check object parameter
        self.assertEqual(params_by_name["object_param"]["type"].__origin__, dict)

        # Check union parameter
        self.assertEqual(params_by_name["union_param"]["type"].__origin__, Union)
        self.assertTrue(str in params_by_name["union_param"]["type"].__args__)
        self.assertTrue(int in params_by_name["union_param"]["type"].__args__)


class TestJsonSchemaToType(unittest.TestCase):
    def test_simple_types(self):
        """Test conversion of simple JSON schema types to Python types."""
        self.assertEqual(json_schema_to_python_type({"type": "string"}), str)
        self.assertEqual(json_schema_to_python_type({"type": "integer"}), int)
        self.assertEqual(json_schema_to_python_type({"type": "number"}), float)
        self.assertEqual(json_schema_to_python_type({"type": "boolean"}), bool)
        self.assertEqual(json_schema_to_python_type({"type": "null"}), type(None))

    def test_array_types(self):
        """Test conversion of array types."""
        # Array of strings
        array_schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        result = json_schema_to_python_type(array_schema)
        self.assertEqual(result.__origin__, list)
        self.assertEqual(result.__args__[0], str)

        # Array of any type
        array_schema = {"type": "array"}
        result = json_schema_to_python_type(array_schema)
        self.assertEqual(result.__origin__, list)
        self.assertEqual(result.__args__[0], Any)

    def test_object_type(self):
        """Test conversion of object type."""
        object_schema = {"type": "object"}
        result = json_schema_to_python_type(object_schema)
        self.assertEqual(result.__origin__, dict)
        self.assertEqual(result.__args__[0], str)
        self.assertEqual(result.__args__[1], Any)

    def test_union_types(self):
        """Test conversion of union types (anyOf/oneOf)."""
        union_schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        }
        result = json_schema_to_python_type(union_schema)
        self.assertEqual(result.__origin__, Union)
        self.assertTrue(str in result.__args__)
        self.assertTrue(int in result.__args__)

    def test_ref_type(self):
        """Test handling of $ref in schema."""
        ref_schema = {"$ref": "#/definitions/SomeType"}
        result = json_schema_to_python_type(ref_schema)
        self.assertEqual(result, Any)

    def test_unknown_type(self):
        """Test handling of unknown schema types."""
        unknown_schema = {"type": "unknown_type"}
        result = json_schema_to_python_type(unknown_schema)
        self.assertEqual(result, Any)
