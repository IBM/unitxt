from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="Human: {source}\nAssistant: {target_prefix}{target}\n\n",
    model_input_format="{system_prompt}{instruction}\n{demos}Human: {source}\nAssistant: {target_prefix}",
)

add_to_catalog(format, "formats.human_assistant", overwrite=True)
