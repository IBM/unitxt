
import unittest
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel
from unitxt.tool_calling import (
    convert_chat_api_format_to_tool,
    convert_to_chat_api_format,
    json_schema_to_python_type,
    pydantic_model_to_canonical_repr,
    tool_to_canonical_representation,
)
from unitxt.types import Tool


class TestConvertToChatAPIFormat(unittest.TestCase):
    def test_basic_conversion_with_description(self):
        """Test basic conversion, including a parameter with a description."""
        tool_input: Tool = {
            "name": "test_tool_desc",
            "description": "A test tool with descriptions",
            "parameters": [
                {"name": "param1", "type": str, "description": "The first parameter."},
                {"name": "param2", "type": int}
            ]
        }
        actual_result = convert_to_chat_api_format(tool_input)

        expected_schema_params = {
            "properties": {
                "param1": {"title": "Param1", "description": "The first parameter.", "type": "string"},
                "param2": {"title": "Param2", "type": "integer"}
            },
            "required": ["param1", "param2"],
            "title": "test_tool_descParams",
            "type": "object"
        }

        expected_result = {
            "type": "function",
            "function": {
                "name": "test_tool_desc",
                "description": "A test tool with descriptions",
                "parameters": expected_schema_params
            }
        }
        self.assertEqual(actual_result, expected_result)

    def test_complex_parameters_with_description(self):
        """Test complex types, including a parameter with a description."""
        tool_input: Tool = {
            "name": "complex_tool_desc",
            "description": "A complex tool with descriptions",
            "parameters": [
                {"name": "list_param", "type": List[str], "description": "A list of strings."},
                {"name": "dict_param", "type": Dict[str, Any]},
                {"name": "any_param", "type": Any, "description": "Can be anything."},
                {"name": "optional_param", "type": Optional[str], "default": None}
            ]
        }
        actual_result = convert_to_chat_api_format(tool_input)
        expected_schema_params = {
            "properties": {
                "list_param": {"title": "List Param", "description": "A list of strings.", "type": "array", "items": {"type": "string"}},
                "dict_param": {"title": "Dict Param", "type": "object"},
                "any_param": {"title": "Any Param", "description": "Can be anything."},
                "optional_param": {
                    "title": "Optional Param",
                    "default": None,
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                }
            },
            "required": ["list_param", "dict_param", "any_param"],
            "title": "complex_tool_descParams",
            "type": "object"
        }
        expected_result = {
            "type": "function",
            "function": {
                "name": "complex_tool_desc",
                "description": "A complex tool with descriptions",
                "parameters": expected_schema_params
            }
        }
        self.assertEqual(actual_result, expected_result)


class TestConvertChatAPIFormatToTool(unittest.TestCase):
    def test_basic_conversion_with_description(self):
        """Test conversion from schema, including a parameter with a description."""
        chat_api_tool = {
            "type": "function",
            "function": {
                "name": "test_tool_from_schema",
                "description": "A test tool from schema with descriptions",
                "parameters": {
                    "title": "TestToolSchemaParams", "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Description for param1 from schema."},
                        "param2": {"type": "integer"}
                    }, "required": ["param1", "param2"]
                }
            }
        }
        actual_tool = convert_chat_api_format_to_tool(chat_api_tool)
        actual_canonical = tool_to_canonical_representation(actual_tool)

        expected_params_canonical = [
            {"name": "param1", "type_name": "str", "description": "Description for param1 from schema."},
            {"name": "param2", "type_name": "int"}
        ]
        expected_canonical = {
            "name": "test_tool_from_schema",
            "description": "A test tool from schema with descriptions",
            "parameters": sorted(expected_params_canonical, key=lambda p: p["name"])
        }
        self.assertEqual(actual_canonical, expected_canonical)

    def test_nested_parameters_conversion_with_description(self):
        """Test nested parameters, including descriptions at various levels."""
        chat_api_tool_nested = {
            "type": "function",
            "function": {
                "name": "nested_tool_desc",
                "description": "A tool with nested parameters and descriptions",
                "parameters": {
                    "title": "NestedToolDescParams", "type": "object",
                    "properties": {
                        "top_level_param": {"type": "string", "description": "Top level string."},
                        "nested_object_param": {
                            "type": "object", "title": "NestedObject", "description": "A nested object.",
                            "properties": {
                                "nested_string": {"type": "string", "description": "String inside nested."},
                                "deeply_nested_object": {
                                    "type": "object", "title": "DeeplyNestedObject", "description": "Even deeper object.",
                                    "properties": {"deep_int": {"type": "integer", "description": "A deep integer."}},
                                    "required": ["deep_int"]
                                }
                            }, "required": ["nested_string"]
                        },
                        "array_of_objects_param": {
                            "type": "array", "description": "Array of items.",
                            "items": {
                                "type": "object", "title": "ArrayItemObject", "description": "Item in an array.",
                                "properties": {
                                    "item_bool": {"type": "boolean", "description": "Boolean item."},
                                    "item_number": {"type": "number"}
                                }, "required": ["item_bool"]
                            }
                        }
                    }, "required": ["top_level_param", "nested_object_param", "array_of_objects_param"]
                }
            }
        }
        actual_tool = convert_chat_api_format_to_tool(chat_api_tool_nested)
        actual_canonical = tool_to_canonical_representation(actual_tool)

        expected_params_list_revised = [
            {"name": "top_level_param", "type_name": "str", "description": "Top level string."},
            {
                "name": "nested_object_param",
                "type_name": "pydantic_model",
                "model_name": "NestedObject",
                "description": "A nested object.",
                "model_fields": {
                    "deeply_nested_object": {
                        "type_name": "pydantic_model",
                        "model_name": "DeeplyNestedObject",
                        "description": "Even deeper object.",
                        "is_required": False,
                        "_is_optional_model_": True, "original_type_structure": "Union_with_None",
                        "model_fields": {
                            "deep_int": {"type_name": "int", "is_required": True, "description": "A deep integer."}
                        }
                    },
                    "nested_string": {"type_name": "str", "is_required": True, "description": "String inside nested."}
                }
            },
            {
                "name": "array_of_objects_param",
                "type_name": "list",
                "description": "Array of items.",
                "item_type": {
                    "type_name": "pydantic_model", "model_name": "ArrayItemObject",
                    "description": "Item in an array.",
                    "model_fields": {
                        "item_bool": {"type_name": "bool", "is_required": True, "description": "Boolean item."},
                        "item_number": {"type_name": "union", "types": sorted(["NoneType", "float"]), "is_required": False}
                    }
                }
            }
        ]

        expected_canonical = {
            "name": "nested_tool_desc",
            "description": "A tool with nested parameters and descriptions",
            "parameters": sorted(expected_params_list_revised, key=lambda p: p["name"])
        }
        self.maxDiff = None
        self.assertEqual(actual_canonical, expected_canonical)


class TestJsonSchemaToType(unittest.TestCase):
    def assert_model_repre(self, schema, expected_repr_dict, model_name_prefix="TestPrefix"):
        model_class = json_schema_to_python_type(schema, model_name_prefix=model_name_prefix)
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
             self.fail(f"Output is not a Pydantic model, got {type(model_class)} for schema {schema}")
        actual_repr = pydantic_model_to_canonical_repr(model_class)
        self.assertEqual(actual_repr, expected_repr_dict)

    def test_simple_types(self):
        self.assertEqual(json_schema_to_python_type({"type": "string"}), str)

    def test_object_type_with_properties_and_field_description(self):
        object_schema = {
            "type": "object", "title": "ObjWithFieldDesc",
            "properties": {
                "name": {"type": "string", "description": "The name field."},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        expected_repr = {
            "type_name": "pydantic_model", "model_name": "ObjWithFieldDesc",
            "model_fields": {
                "age": {"type_name": "union", "types": sorted(["NoneType", "int"]), "is_required": False},
                "name": {"type_name": "str", "is_required": True, "description": "The name field."}
            }
        }
        self.assert_model_repre(object_schema, expected_repr, model_name_prefix="TestFieldDesc")


    def test_optional_string_from_type_list(self):
        schema = {"type": ["string", "null"]}
        self.assertEqual(json_schema_to_python_type(schema), Optional[str])

    def test_integer_type(self):
        self.assertEqual(json_schema_to_python_type({"type": "integer"}), int)

    def test_number_type(self):
        self.assertEqual(json_schema_to_python_type({"type": "number"}), float)

    def test_boolean_type(self):
        self.assertEqual(json_schema_to_python_type({"type": "boolean"}), bool)

    def test_null_type(self):
        self.assertEqual(json_schema_to_python_type({"type": "null"}), type(None))

    def test_array_of_strings(self):
        self.assertEqual(json_schema_to_python_type({"type": "array", "items": {"type": "string"}}), List[str])

    def test_array_of_any(self):
        self.assertEqual(json_schema_to_python_type({"type": "array"}), List[Any])

    def test_array_of_empty_items(self):
        self.assertEqual(json_schema_to_python_type({"type": "array", "items": {}}), List[Any])

    def test_object_type_no_properties(self):
        self.assertEqual(json_schema_to_python_type({"type": "object"}), Dict[str, Any])

    def test_object_type_empty_properties(self):
        self.assertEqual(json_schema_to_python_type({"type": "object", "properties": {}}), Dict[str, Any])

    def test_union_types_simple(self):
        actual = json_schema_to_python_type({"anyOf": [{"type": "string"}, {"type": "integer"}]})
        self.assertEqual(actual, Union[int, str])

    def test_ref_type(self):
        self.assertEqual(json_schema_to_python_type({"$ref": "#/definitions/SomeType"}), Any)

    def test_unknown_type(self):
        self.assertEqual(json_schema_to_python_type({"type": "unknown_type"}), Any)

