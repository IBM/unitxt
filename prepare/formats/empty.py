from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="{source}\\N{target_prefix}{target}\n\n",
    model_input_format="{system_prompt}\\N{instruction}\\N{demos}{source}\\N{target_prefix}",
)

add_to_catalog(format, "formats.empty", overwrite=True)
