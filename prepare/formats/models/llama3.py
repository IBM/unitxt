from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

# see: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
# According to: https://huggingface.co/blog/llama3#how-to-prompt-llama-3
# The Instruct versions use the following conversation structure:
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
#
# {{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# {{ model_answer_1 }}<|eot_id|>

format = SystemFormat(
    demo_format="<|start_header_id|>user<|end_header_id|>\n\n"
    "{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{target_prefix}{target}<|eot_id|>",
    model_input_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    + "{system_prompt}{instruction}"
    + "<|eot_id|>{demos}<|start_header_id|>user<|end_header_id|>\n\n"
    "{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_prefix}",
)

add_to_catalog(
    format,
    "formats.llama3_instruct",
    overwrite=True,
)

format = SystemFormat(
    demo_format="{source}\n\n{target_prefix}{target}\n\n",
    model_input_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}{instruction}"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{demos}"
    "{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_prefix}",
)

add_to_catalog(
    format,
    "formats.llama3_instruct_all_demos_in_one_turn",
    overwrite=True,
)
