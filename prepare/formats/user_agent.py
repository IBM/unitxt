from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="User: {source}\nAgent: {target_prefix}{target}\n\n",
    model_input_format="{system_prompt}{instruction}\n{demos}\nUser:{source}\nAgent:{target_prefix}",
)

add_to_catalog(format, "formats.user_agent", overwrite=True)
