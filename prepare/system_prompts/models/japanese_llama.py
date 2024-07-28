from unitxt.catalog import add_to_catalog
from unitxt.system_prompts import TextualSystemPrompt

system_prompt = TextualSystemPrompt(
    "<<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n"
)

add_to_catalog(
    system_prompt,
    "system_prompts.models.japanese_llama",
    overwrite=True,
)
