from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="{source}\n {target_prefix}{target}\n\n",
    model_input_format="{system_prompt}{instruction}{demos}{source}\n{target_prefix}",
)

add_to_catalog(format, "formats.empty", overwrite=True)
