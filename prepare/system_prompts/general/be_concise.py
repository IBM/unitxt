from unitxt.catalog import add_to_catalog
from unitxt.system_prompts import TextualSystemPrompt

system_prompt = TextualSystemPrompt(
    "be concise. at every point give the shortest acceptable answer."
)
add_to_catalog(system_prompt, "system_prompts.general.be_concise", overwrite=True)
