from src.unitxt.catalog import add_to_catalog
from src.unitxt.system_prompts import TextualSystemPrompt

system_prompt = TextualSystemPrompt(
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
)

add_to_catalog(system_prompt, "system_prompts.models.alpaca", overwrite=True)
