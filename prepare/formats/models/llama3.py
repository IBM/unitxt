from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

# see: https://huggingface.co/blog/llama3#how-to-prompt-llama-3

format = SystemFormat(
    model_input_format="<|start_header_id|>system<|end_header_id|>\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{instruction}\\N{source}\\N{target_prefix}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n",
)

add_to_catalog(format, "formats.models.llama3", overwrite=True)
