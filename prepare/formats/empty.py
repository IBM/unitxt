from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="{source}\n {target}\n\n",
    model_input_format="{instruction}{demos}{source}\n",
)

add_to_catalog(format, "formats.empty", overwrite=True)
