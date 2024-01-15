from unitxt.catalog import add_to_catalog
from unitxt.instructions import TextualInstruction

instruction = TextualInstruction(
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
)

add_to_catalog(instruction, "instructions.models.alpaca", overwrite=True)
