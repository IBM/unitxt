from unitxt.catalog import add_to_catalog
from unitxt.system_prompts import TextualSystemPrompt

# DeepSeek-Coder format and system prompt according to: https://github.com/deepseek-ai/deepseek-coder

system_prompt = TextualSystemPrompt(
    "You are an AI programming assistant, utilizing the DeepSeek Coder "
    "model, developed by DeepSeek Company, and you only answer questions "
    "related to computer science. For politically sensitive questions, "
    "security and privacy issues, and other non-computer science questions, "
    "you will refuse to answer."
)
add_to_catalog(system_prompt, "system_prompts.models.deepseek_coder", overwrite=True)
