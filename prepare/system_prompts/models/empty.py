from src.unitxt.catalog import add_to_catalog
from src.unitxt.system_prompts import TextualSystemPrompt

system_prompt = TextualSystemPrompt("")

add_to_catalog(system_prompt, "system_prompts.empty", overwrite=True)
