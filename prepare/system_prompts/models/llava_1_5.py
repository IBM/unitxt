from unitxt.catalog import add_to_catalog
from unitxt.system_prompts import TextualSystemPrompt

system_prompt = TextualSystemPrompt(
    "You are a helpful assistant."
)
add_to_catalog(system_prompt, "system_prompts.models.llava1_5", overwrite=True)
