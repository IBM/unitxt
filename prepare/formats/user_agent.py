from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import SystemFormat

format = SystemFormat(
    demo_format="User: {source}\nAgent: {target}\n\n",
    model_input_format="{instruction}\n{demos}\nUser:{source}\nAgent:",
)

add_to_catalog(format, "formats.user_agent", overwrite=True)
