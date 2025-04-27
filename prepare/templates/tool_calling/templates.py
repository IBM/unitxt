from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(input_format="{query}", output_format="{call}", postprocessors=["processors.load_json_or_empty_tool_call"]),
    "templates.tool_calling.base",
    overwrite=True,
)
