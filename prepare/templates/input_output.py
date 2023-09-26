from src.unitxt.blocks import InputOutputTemplate
from src.unitxt.catalog import add_to_catalog

template = InputOutputTemplate(
    input_format="{input}",
    output_format="{target}",
)
add_to_catalog(template, f"templates.input_output", overwrite=True)
