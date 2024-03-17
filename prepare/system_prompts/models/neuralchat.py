from src.unitxt.catalog import add_to_catalog
from src.unitxt.system_prompts import TextualSystemPrompt

system_prompt = TextualSystemPrompt(
    "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n"
)

add_to_catalog(system_prompt, "system_prompts.models.neuralchat", overwrite=True)
