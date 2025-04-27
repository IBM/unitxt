from typing import Any, Dict, List, Type

from .operators import FieldOperator
from .types import Parameter, Tool


def convert_to_chat_api_format(tool: Tool) -> Dict[str, Any]:

    from pydantic import create_model

    field_definitions = {}
    for param in tool["parameters"]:
        param_name = param["name"]
        param_type = param.get("type", Any)
        field_definitions[param_name] = (param_type, ...)  # ... means required in Pydantic

    model = create_model(f"{tool['name']}Params", **field_definitions)

    schema = model.model_json_schema()

    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": schema
        }
    }


def convert_chat_api_format_to_tool(chat_api_tool: Dict[str, Any]) -> Tool:
    """Convert a Chat API formatted tool back to the original Tool structure.

    Args:
        chat_api_tool: A dictionary representing a tool in Chat API format

    Returns:
        A Tool dictionary with name, description, and parameters
    """
    # Extract function information
    function_info = chat_api_tool.get("function", {})
    name = function_info.get("name", chat_api_tool.get("name", ""))
    description = function_info.get("description", chat_api_tool.get("description", ""))

    # Extract parameters from schema
    parameters: List[Parameter] = []
    schema = function_info.get("parameters",  chat_api_tool.get("parameters", ""))
    properties = schema.get("properties", {})

    for param_name, param_schema in properties.items():
        # Map JSON schema type to Python type
        param_type = json_schema_to_python_type(param_schema)

        parameter: Parameter = {
            "name": param_name,
            "type": param_type
        }
        parameters.append(parameter)

    # Construct and return the Tool
    tool: Tool = {
        "name": name,
        "description": description,
        "parameters": parameters
    }

    return tool

def json_schema_to_python_type(schema: Dict[str, Any]) -> Type:
    """Convert JSON schema type to Python type."""
    from typing import Any, Dict, List, Union

    schema_type = schema.get("type")

    # Handle simple types
    simple_types = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None)
    }

    if schema_type in simple_types:
        return simple_types[schema_type]

    # Handle arrays
    if schema_type == "array":
        items = schema.get("items", {})
        if not items:
            return List[Any]

        item_type = json_schema_to_python_type(items)
        return List[item_type]

    # Handle objects
    if schema_type == "object":
        return Dict[str, Any]

    # Handle unions with anyOf/oneOf
    if "anyOf" in schema or "oneOf" in schema:
        union_schemas = schema.get("anyOf", []) or schema.get("oneOf", [])
        union_types = [json_schema_to_python_type(s) for s in union_schemas]
        # Use Union for Python 3.9+ or create Union using typing module
        return Union[tuple(union_types)] if union_types else Any

    # Handle references (simplified)
    if "$ref" in schema:
        # In a real implementation, you'd resolve references
        return Any

    # Default to Any for unrecognized schema types
    return Any


class ToTool(FieldOperator):

    def process_value(self, value: Dict[str, Any]) -> Tool:
        return convert_chat_api_format_to_tool(value)
