from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="{source}\n{target_prefix}{target}\n\n",
    model_input_format="[INST] {system_prompt}\n{instruction}\n{demos}\n{source}\n[/INST]{target_prefix}",
)

add_to_catalog(format, "formats.llama", overwrite=True)
