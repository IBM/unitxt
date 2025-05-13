from unitxt import add_to_catalog
from unitxt.processors import (
    PostProcess,
)
from unitxt.struct_data_operators import ToolCallPostProcessor

add_to_catalog(
    PostProcess(ToolCallPostProcessor(allow_failure=True, failure_value={"name": "null", "arguments": {}})),
    "processors.load_json_or_empty_tool_call",
    overwrite=True,
)
