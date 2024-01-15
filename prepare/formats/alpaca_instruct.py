from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="### Instruction:\n{source}\n\n\n### Response: {target}\n\n",
    model_input_format="{instruction}{demos}### Instruction:\n{source}\n\n\n### Response: ",
)

add_to_catalog(format, "formats.alpaca_instruct", overwrite=True)
