from unitxt.catalog import add_to_catalog
from unitxt.instructions import TextualInstruction

instruction = TextualInstruction(
    "<<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n"
)

add_to_catalog(
    instruction,
    "instructions.models.japanese_llama",
    overwrite=True,
)
