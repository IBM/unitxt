from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

# See https://huggingface.co/Nexusflow/Starling-LM-7B-beta


format = SystemFormat(
    demo_format="{source}\n\n{target_prefix}{target}\n\n",
    model_input_format="GPT4 Correct User: {instruction}{demos}\\N{source}<|end_of_turn|>"
    "GPT4 Correct Assistant: {target_prefix}",
)

add_to_catalog(format, "formats.models.starling", overwrite=True)
