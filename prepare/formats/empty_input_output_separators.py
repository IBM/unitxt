from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="{source}{target}\n\n",
    model_input_format="{instruction}\n{demos}\n{source}",
)

add_to_catalog(format, "formats.empty_input_output_separator", overwrite=True)
