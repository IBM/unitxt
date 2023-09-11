from src.unitxt.catalog import add_to_catalog
from src.unitxt.instructions import TextualInstruction

instruction = TextualInstruction("You are a very smart model. solve the following.")

add_to_catalog(instruction, f"instructions.models.llama", overwrite=True)
