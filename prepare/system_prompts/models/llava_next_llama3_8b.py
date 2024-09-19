from unitxt.catalog import add_to_catalog
from unitxt.system_prompts import TextualSystemPrompt

# See https://huggingface.co/docs/transformers/en/model_doc/llava_next

system_prompt = TextualSystemPrompt(
    "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
)

add_to_catalog(
    system_prompt, "system_prompts.models.llava_next_llama3_8b", overwrite=True
)
