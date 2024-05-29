from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

# see: https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/inference/prompt_format_utils.py

# PROMPT_TEMPLATE_2 = Template(f"[INST] {PROMPT_TASK}{SAFETY_CATEGORIES}{PROMPT_CONVERSATION}{PROMPT_INSTRUCTIONS_2} [/INST]")

format = SystemFormat(
    demo_format="",  # User: {question}\n\nAgent: {answer}\n\n",
    model_input_format="[INST] {source} [/INST]",
)

add_to_catalog(format, "formats.llamaguard2", overwrite=True)
