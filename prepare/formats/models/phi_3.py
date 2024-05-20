from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

format = SystemFormat(
    demo_format="<|user|>\n{instruction}{source}<|end|>\n"
    "<|assistant|>\n{target_prefix}{target}<|end|>\n",
    model_input_format="<|user|>\n{system_prompt}<|end|>\n"
    "{demos}"
    "<|user|>\n{instruction}{source}<|end|>\n"
    "<|assistant|>\n{target_prefix}",
)

add_to_catalog(format, "formats.models.phi_3", overwrite=True)
