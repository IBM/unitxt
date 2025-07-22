from unitxt.catalog import add_to_catalog
from unitxt.operators import ExecuteExpression

# Normalize tool schema by converting "arguments" to "parameters" field names
# This handles inconsistent tool definitions where some use "arguments" and others use "parameters"
operator = ExecuteExpression(
    expression='[{("parameters" if k == "arguments" else k): v for k, v in tool.items()} for tool in tools]',
    to_field="tools",
)

add_to_catalog(
    operator,
    "operators.normalize_tool_schema",
    overwrite=True,
)
