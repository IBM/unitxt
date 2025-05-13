from unitxt.catalog import add_to_catalog
from unitxt.templates import MultiReferenceTemplate

add_to_catalog(
    MultiReferenceTemplate(
        input_format="{query}",
        references_field="reference_calls",
        postprocessors=["processors.load_json_or_empty_tool_call"],
    ),
    "templates.tool_calling.base",
    overwrite=True,
)
