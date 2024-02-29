from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="### Instruction:\n{source}\n\n\n### Response: {target}\n\n",
    model_input_format="{system_prompt}{demos}### Instruction:\n{source}\n\n\n### Response: ",
)

add_to_catalog(format, "formats.models.alpaca_instruct", overwrite=True)
