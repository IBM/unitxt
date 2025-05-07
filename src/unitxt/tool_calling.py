import keyword
from typing import Any, Dict, List, Type, Union, get_args, get_origin

from .operators import FieldOperator
from .types import Parameter, ParameterWithDescription, Tool

try:
    from pydantic import BaseModel, Field, create_model
    from pydantic_core import PydanticUndefined
except:
    pass

def convert_to_chat_api_format(tool: Tool) -> Dict[str, Any]:
    """Converts a Tool object (Python TypedDict representation) into the OpenAI Chat API JSON Schema format.

    Includes parameter descriptions if provided and handles optional parameters
    if a 'default' key is present in their definition.
    Sanitizes field names that start with underscores for Pydantic compatibility.
    """
    field_definitions = {}
    for param in tool["parameters"]:
        original_param_name = param["name"]
        param_type = param["type"]

        param_description = param.get("description")

        field_args = {}
        if param_description is not None:
            field_args["description"] = param_description

        python_field_name = original_param_name
        if original_param_name.startswith("_"):
            field_args["alias"] = original_param_name

            stripped_name_part = original_param_name.lstrip("_")
            if not stripped_name_part:
                python_field_name = "p_field"
            else:

                potential_python_name = "p_" + stripped_name_part
                if potential_python_name.isidentifier():
                    python_field_name = potential_python_name
                else:
                    sanitized_suffix = "".join(c if c.isalnum() or c == "_" else "_" for c in stripped_name_part)
                    python_field_name = "p_s_" + sanitized_suffix # "p_s_" for "p_sanitized_"
                    if python_field_name == "p_s_":
                        python_field_name = "p_s_field_fallback"
                    if not python_field_name.isidentifier():
                        python_field_name = f"p_aliased_hash_{abs(hash(original_param_name))}"

        if "default" in param:
            field_args["default"] = param["default"]
            pydantic_field_instance = Field(**field_args)
        else:
            pydantic_field_instance = Field(**field_args, default=...)

        field_definitions[python_field_name] = (param_type, pydantic_field_instance)


    model_name_base = "".join(c if c.isalnum() else "_" for c in str(tool["name"]))
    if not model_name_base or model_name_base[0].isdigit():
        model_name_base = "Model_" + model_name_base
    model_name = model_name_base + "Params"

    if not field_definitions:
        model_for_schema = create_model(model_name)
    else:
        model_for_schema = create_model(model_name, **field_definitions)

    schema = model_for_schema.model_json_schema()

    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": schema
        }
    }


def json_schema_to_python_type(schema: Dict[str, Any], model_name_prefix: str = "DynamicModel") -> Type:
    """Converts a JSON schema type definition into a Python type.

    Handles simple types, arrays, objects (creating Pydantic models), and unions.
    Propagates 'description' for fields within generated Pydantic models and
    attaches schema block descriptions to generated models.
    """
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        py_types = []
        has_null = "null" in schema_type
        for st_item in schema_type:
            if st_item == "null":
                continue
            py_types.append(json_schema_to_python_type({"type": st_item}, model_name_prefix))

        py_types = [pt for pt in py_types if pt is not Any]

        if not py_types:
            return type(None) if has_null else Any

        core_type = Union[tuple(sorted(py_types, key=str))] if len(py_types) > 1 else py_types[0]

        if has_null:
            if core_type is type(None):
                return type(None)
            if get_origin(core_type) is Union and type(None) in get_args(core_type):
                return core_type
            return Union[core_type, type(None)]
        return core_type

    simple_types_map = {
        "string": str, "integer": int, "number": float, "boolean": bool, "null": type(None)
    }
    if schema_type in simple_types_map:
        return simple_types_map[schema_type]

    if schema_type == "array":
        items_schema = schema.get("items", {})
        if not items_schema:
            return List[Any]
        item_title = items_schema.get("title", "Item") if isinstance(items_schema, dict) else "Item"
        item_model_name = f"{model_name_prefix}{item_title.capitalize().replace(' ','')}"

        item_type = json_schema_to_python_type(items_schema, model_name_prefix=item_model_name)
        return List[item_type]

    if schema_type == "object":
        properties = schema.get("properties", {})
        object_title = schema.get("title")
        object_description = schema.get("description")

        current_model_name_base = object_title if (object_title and isinstance(object_title, str) and object_title.strip()) else model_name_prefix
        current_model_name = "".join(c for c in current_model_name_base if c.isalnum() or c == "_")
        if not current_model_name or current_model_name[0].isdigit():
            current_model_name = "Model_" + current_model_name
        if not current_model_name:
            current_model_name = "UnnamedDynamicModel"

        if not properties:
            return Dict[str, Any]

        field_definitions = {}
        required_fields = schema.get("required", [])
        for prop_name, prop_schema in properties.items():
            original_prop_name = prop_name
            python_field_name = original_prop_name

            field_args = {}
            if original_prop_name.startswith("_"):
                python_field_name_candidate = "p_" + original_prop_name.lstrip("_")
                if not python_field_name_candidate.isidentifier() or python_field_name_candidate == "p_":
                    sanitized_suffix = "".join(c if c.isalnum() else "_" for c in original_prop_name.lstrip("_"))
                    python_field_name_candidate = "p_s_" + (sanitized_suffix or "field")
                    if not python_field_name_candidate.isidentifier():
                        python_field_name_candidate = f"p_aliased_hash_{abs(hash(original_prop_name))}"
                field_args["alias"] = original_prop_name
                python_field_name = python_field_name_candidate
            elif not python_field_name.isidentifier() or keyword.iskeyword(python_field_name):
                 sanitized_name = "".join(c if c.isalnum() else "_" for c in python_field_name)
                 python_field_name_candidate = f"field_{sanitized_name}"
                 if not python_field_name_candidate.isidentifier():
                     python_field_name_candidate = f"field_hash_{abs(hash(original_prop_name))}"
                 field_args["alias"] = original_prop_name
                 python_field_name = python_field_name_candidate

            nested_item_title = prop_schema.get("title", original_prop_name.capitalize()) if isinstance(prop_schema, dict) else original_prop_name.capitalize()
            nested_model_name_hint = f"{current_model_name}_{nested_item_title.replace(' ','')}"

            field_type = json_schema_to_python_type(prop_schema, model_name_prefix=nested_model_name_hint)
            is_prop_required = original_prop_name in required_fields

            prop_description = prop_schema.get("description")
            if isinstance(prop_description, str) and prop_description.strip():
                field_args["description"] = prop_description

            pydantic_field = Field(**field_args, default=... if is_prop_required else None)

            field_definitions[python_field_name] = (field_type if is_prop_required else Union[field_type, None], pydantic_field)

        if not field_definitions:
            return Dict[str, Any]
        try:
            created_model = create_model(current_model_name, **field_definitions)
            if isinstance(object_description, str) and object_description.strip():
                created_model._schema_block_description = object_description
            return created_model
        except Exception:
            return Dict[str, Any]

    if "anyOf" in schema or "oneOf" in schema:
        union_schemas = schema.get("anyOf", []) or schema.get("oneOf", [])
        union_types = []
        for i, s_option in enumerate(union_schemas):
            option_title = s_option.get("title", f"Option{i}") if isinstance(s_option, dict) else f"Option{i}"
            option_model_name = f"{model_name_prefix}{option_title.capitalize().replace(' ','')}"
            union_types.append(json_schema_to_python_type(s_option, model_name_prefix=option_model_name))

        has_none = any(ut is type(None) for ut in union_types)
        distinct_types = sorted({ut for ut in union_types if ut is not Any and ut is not type(None)}, key=str)

        if not distinct_types:
            return type(None) if has_none else Any
        core_union_type = Union[tuple(distinct_types)] if len(distinct_types) > 1 else distinct_types[0]
        return Union[core_union_type, type(None)] if has_none and core_union_type is not type(None) else core_union_type

    if "$ref" in schema:
        return Any
    return Any


def convert_chat_api_format_to_tool(chat_api_tool: Dict[str, Any]) -> Tool:
    """Converts a tool definition from OpenAI Chat API JSON Schema format back into a Python Tool TypedDict.

    Includes parameter descriptions if present in the schema.
    """
    function_info = chat_api_tool.get("function", {})
    name = function_info.get("name", chat_api_tool.get("name", ""))
    description = function_info.get("description", chat_api_tool.get("description", ""))

    parameters_list: List[Union[Parameter, ParameterWithDescription]] = []

    tool_name_cleaned = "".join(c if c.isalnum() else "_" for c in str(name))
    if not tool_name_cleaned or tool_name_cleaned[0].isdigit():
        tool_name_cleaned = "Tool"

    schema = function_info.get("parameters", chat_api_tool.get("parameters", {}))
    properties = schema.get("properties", {})

    for param_name, param_schema_item in properties.items():
        param_item_title = param_schema_item.get("title", param_name.capitalize()) if isinstance(param_schema_item, dict) else param_name.capitalize()
        model_prefix_for_param_type = f"{tool_name_cleaned}Params_{param_item_title.replace(' ','')}"

        param_type = json_schema_to_python_type(param_schema_item, model_name_prefix=model_prefix_for_param_type)
        param_description_from_schema = param_schema_item.get("description")

        if isinstance(param_description_from_schema, str) and param_description_from_schema.strip():
            parameter_entry: ParameterWithDescription = {
                "name": param_name,
                "type": param_type,
                "description": param_description_from_schema
            }
        else:
            parameter_entry: Parameter = {
                "name": param_name,
                "type": param_type
            }
        parameters_list.append(parameter_entry)

    tool_result: Tool = {"name": name, "description": description, "parameters": parameters_list}
    return tool_result

def pydantic_model_to_canonical_repr(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Generates a canonical dictionary representation of a Pydantic model's structure."""
    fields_repr = {}
    sorted_field_names = sorted(model_class.model_fields.keys())

    model_level_description = getattr(model_class, "_schema_block_description", None)

    for field_name in sorted_field_names:
        field_info = model_class.model_fields[field_name]
        field_type_annotation = field_info.annotation

        is_field_required = (field_info.default is PydanticUndefined and
                             field_info.default_factory is None)

        canonical_field_key = field_info.alias if field_info.alias else field_name

        field_entry: Dict[str, Any] = {"is_required": is_field_required}

        if field_info.description:
            field_entry["description"] = field_info.description

        origin_field_type = get_origin(field_type_annotation)
        args_field_type = get_args(field_type_annotation)

        if isinstance(field_type_annotation, type) and issubclass(field_type_annotation, BaseModel):
            nested_model_repr = pydantic_model_to_canonical_repr(field_type_annotation)
            field_entry.update(nested_model_repr)
        elif origin_field_type is Union and any(isinstance(arg, type) and issubclass(arg, BaseModel) for arg in args_field_type if arg is not type(None)):
            nested_model_type = next(arg for arg in args_field_type if isinstance(arg, type) and issubclass(arg, BaseModel))
            nested_model_repr = pydantic_model_to_canonical_repr(nested_model_type)
            field_entry.update(nested_model_repr)
            if type(None) in args_field_type:
                 field_entry["_is_optional_model_"] = True
                 field_entry["original_type_structure"] = "Union_with_None"
        elif origin_field_type is list:
            field_entry["type_name"] = "list"
            if args_field_type and args_field_type[0] is not Any:
                list_item_type = args_field_type[0]
                if isinstance(list_item_type, type) and issubclass(list_item_type, BaseModel):
                     field_entry["item_type"] = pydantic_model_to_canonical_repr(list_item_type)
                elif get_origin(list_item_type) is Union:
                     item_union_types = sorted([t.__name__ if hasattr(t, "__name__") else str(t) for t in get_args(list_item_type)])
                     field_entry["item_type"] = {"type_name": "union", "types": item_union_types}
                elif isinstance(list_item_type, type):
                     field_entry["item_type"] = {"type_name": list_item_type.__name__}
                else:
                     field_entry["item_type"] = {"type_name": "Any" if list_item_type is Any else str(list_item_type)}
            else:
                field_entry["item_type"] = {"type_name": "Any"}
        elif origin_field_type is Union:
            type_names = sorted([t.__name__ if hasattr(t, "__name__") else str(t) for t in args_field_type])
            field_entry["type_name"] = "union"
            field_entry["types"] = type_names
        elif isinstance(field_type_annotation, type):
            field_entry["type_name"] = field_type_annotation.__name__
        elif field_type_annotation is Any:
            field_entry["type_name"] = "Any"
        else:
            field_entry["type_name"] = str(field_type_annotation)

        fields_repr[canonical_field_key] = field_entry

    model_repr = {
        "type_name": "pydantic_model",
        "model_name": model_class.__name__,
        "model_fields": fields_repr
    }
    if model_level_description:
        model_repr["description"] = model_level_description
    return model_repr

def tool_to_canonical_representation(tool_with_dynamic_types: Tool) -> Dict[str, Any]:
    """Converts a Tool object into a canonical dictionary representation for tests."""
    canonical_params = []
    for param in tool_with_dynamic_types.get("parameters", []):
        param_type = param["type"]
        canonical_param: Dict[str, Any] = {"name": param["name"]}

        description = param.get("description")
        if description is not None:
            canonical_param["description"] = description

        origin_type = get_origin(param_type)
        args_type = get_args(param_type)

        if isinstance(param_type, type) and issubclass(param_type, BaseModel):
            canonical_param.update(pydantic_model_to_canonical_repr(param_type))
        elif origin_type is list:
            canonical_param["type_name"] = "list"
            if args_type and args_type[0] is not Any :
                list_item_type = args_type[0]

                if isinstance(list_item_type, type) and issubclass(list_item_type, BaseModel):
                    item_model_repr = pydantic_model_to_canonical_repr(list_item_type)
                    canonical_param["item_type"] = item_model_repr
                elif get_origin(list_item_type) is Union:
                    item_union_types = sorted([t.__name__ if hasattr(t, "__name__") else str(t) for t in get_args(list_item_type)])
                    canonical_param["item_type"] = {"type_name": "union", "types": item_union_types}
                elif isinstance(list_item_type, type):
                    canonical_param["item_type"] = {"type_name": list_item_type.__name__}
                else:
                     canonical_param["item_type"] = {"type_name": str(list_item_type)}
            else:
                canonical_param["item_type"] = {"type_name": "Any"}
        elif origin_type is Union:
            type_names = sorted([t.__name__ if hasattr(t, "__name__") else str(t) for t in args_type])
            canonical_param["type_name"] = "union"
            canonical_param["types"] = type_names
        elif origin_type is dict and args_type:
            canonical_param["type_name"] = "Dict"
            canonical_param["args_type_names"] = [t.__name__ if hasattr(t, "__name__") else str(t) for t in args_type]
        elif isinstance(param_type, type):
            canonical_param["type_name"] = param_type.__name__
        elif param_type is Any:
            canonical_param["type_name"] = "Any"
        else:
            canonical_param["type_name"] = str(param_type)

        canonical_params.append(canonical_param)

    return {
        "name": tool_with_dynamic_types["name"],
        "description": tool_with_dynamic_types["description"],
        "parameters": sorted(canonical_params, key=lambda p: p["name"])
    }

class ToTool(FieldOperator):

    def process_value(self, value: Dict[str, Any]) -> Tool:
        return convert_chat_api_format_to_tool(value)
