from unitxt.catalog import add_to_catalog
from unitxt.operators import RecursiveReplace

operator = RecursiveReplace(
    key="type",
    map_values={
        "dict": "object",
        "float": "number",
        "tuple": "array",
        "HashMap": "object",
        "bool": "boolean",
        "list": "array",
        "any": "string",
        "int": "integer",
        "byte": "integer",
        "short": "integer",
        "long": "integer",
        "double": "number",
        "char": "string",
        "ArrayList": "array",
        "Array": "array",
        "Hashtable": "object",
        "Queue": "array",
        "Stack": "array",
        "Any": "string",
        "String": "string",
        "str, optional": "string",
        "str": "string",
        "Bigint": "integer",
        "Set": "array",
        "Boolean": "boolean",
    },
    remove_values=["any"],
)

add_to_catalog(
    operator,
    "operators.fix_json_schema",
    overwrite=True,
)
