from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="### Prompt:\n{source}\n### Response:\n {target_prefix}{target}\n\n",
    model_input_format="{system_prompt}{instruction}{demos}### Prompt:\n{source}\n### Response:\n{target_prefix}",
)

add_to_catalog(format, "formats.###prompt_###response", overwrite=True)
