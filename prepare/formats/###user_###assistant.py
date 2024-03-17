from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="### User:\n{source}\n### Assistant:\n {target_prefix}{target}\n\n",
    model_input_format="{system_prompt}{instruction}{demos}### User:\n{source}\n### Assistant:\n{target_prefix}",
)

add_to_catalog(format, "formats.###user_###assistant", overwrite=True)
