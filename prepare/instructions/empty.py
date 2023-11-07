from src.unitxt.catalog import add_to_catalog
from src.unitxt.instructions import TextualInstruction

instruction = TextualInstruction("")
add_to_catalog(instruction, f"instructions.empty", overwrite=True)
