from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="{source}{target_prefix}{target}\n\n",
    model_input_format="{system_prompt}{instruction}\n{demos}\n{source}{target_prefix}",
)

add_to_catalog(format, "formats.empty_input_output_separator", overwrite=True)
