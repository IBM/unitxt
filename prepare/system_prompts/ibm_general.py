from src.unitxt.catalog import add_to_catalog
from src.unitxt.system_prompts import TextualSystemPrompt

system_prompt = TextualSystemPrompt(
    "### System:\nYou are a helpful AI language model developed by IBM. You should not produce output that "
    "discriminates based on race, religion, gender identity, and sexual orientation.\n "
)

add_to_catalog(system_prompt, "system_prompts.ibm_general", overwrite=True)
