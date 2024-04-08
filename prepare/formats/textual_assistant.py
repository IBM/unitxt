from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="<|user|>\n{source}\n<|assistant|>\n{target_prefix}{target}\n",
    model_input_format="{system_prompt}\n{instruction}\n{demos}<|user|>\n{source}\n<|assistant|>\n{target_prefix}",
)

add_to_catalog(format, "formats.textual_assistant", overwrite=True)
