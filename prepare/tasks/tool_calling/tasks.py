from typing import List

from unitxt.catalog import add_to_catalog
from unitxt.task import Task
from unitxt.types import Tool, ToolCall, Turn

add_to_catalog(
    Task(
        __description__="""Task to test tool calling capabilities.  It assume the model is provided with a query and is requested to invoke a single tool from the list of provided tools.

        Reference_calls is a list of ground truth tool calls to compare with.
        """,
        input_fields={"query": str, "tools": List[Tool]},
        reference_fields={"reference_calls": List[ToolCall]},
        prediction_type=ToolCall,
        metrics=["metrics.tool_calling"],
        default_template="templates.tool_calling.base",
        requirements=["jsonschema-rs"],
    ),
    "tasks.tool_calling.supervised",
    overwrite=True,
)

add_to_catalog(
    Task(
        __description__="""Task to test tool calling capabilities.  It assume the model is provided with a dialog (set of turns) and is requested to invoke a single tool from the list of provided tools given the dialog.

        Reference_calls is a list of ground truth tool calls to compare with.
        """,
        input_fields={"dialog": List[Turn], "tools": List[Tool]},
        reference_fields={"reference_calls": List[ToolCall]},
        prediction_type=List[ToolCall],
        metrics=[
            "metrics.tool_calling.multi_turn.validity",
        ],
        default_template="templates.tool_calling.multi_turn",
        requirements=["jsonschema-rs"],
    ),
    "tasks.tool_calling.multi_turn",
    overwrite=True,
)
