from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import SystemFormat

format = SystemFormat(
    demo_format="{source}\n{target}\n\n",
    model_input_format="[INST] {instruction}\n{demos}\n{source}\n[/INST]",
)

add_to_catalog(format, "formats.llama", overwrite=True)
