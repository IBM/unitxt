from typing import List

from unitxt.catalog import add_to_catalog
from unitxt.task import Task
from unitxt.types import Tool, ToolCall

add_to_catalog(
    Task(
        __description__="""Task to test tool calling capabilities.""",
        input_fields={"query": str, "tools": List[Tool]},
        reference_fields={"reference_calls": List[ToolCall]},
        prediction_type=ToolCall,
        metrics=["metrics.tool_calling"],
        default_template="templates.tool_calling.base",
    ),
    "tasks.tool_calling.supervised",
    overwrite=True,
)
