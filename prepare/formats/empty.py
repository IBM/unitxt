from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="{source} {target_prefix}{target}\n\n",
    model_input_format="{system_prompt}\\N{instruction}\\N{demos}{source}{target_prefix}",
)

default_format = SystemFormat()
"""
assert (
    format.demo_format == default_format.demo_format
), f"{format.demo_format} != {default_format.demo_format}"
assert (
    format.model_input_format == default_format.model_input_format
), f"{format.model_input_format} != {default_format.model_input_format}"
"""

add_to_catalog(format, "formats.empty", overwrite=True)
