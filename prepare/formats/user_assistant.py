from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="<|user|>\n{source}\n<|assistant|>\n {target}\n\n",
    model_input_format="{instruction}{demos}<|user|>\n{source}\n<|assistant|>\n",
)

add_to_catalog(format, "formats.user_assistant", overwrite=True)
