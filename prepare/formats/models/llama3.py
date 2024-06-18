from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

# see: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
# {{ user_message }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

format = SystemFormat(
    demo_format="{source}\n\n{target_prefix}{target}\n\n",
    model_input_format="<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{instruction}{demos}{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    "{target_prefix}",
)

add_to_catalog(format, "formats.llama3_chat", overwrite=True)

format = SystemFormat(
    __description__=""" Format used by Llama3 models in chat mode, where demos are added in the first utterence""",
    demo_format="{source}\n\n{target_prefix}{target}\n\n",
    model_input_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{instruction}{demos}{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    "{target_prefix}",
)

add_to_catalog(format, "formats.llama3_chat_with_system_prompt", overwrite=True)

format = SystemFormat(
    __description__=""" Format used by Llama3 models in multi chat mode, where no demos are used and previous dialog turns are added
    as part of the source field using the demo_format with SerializeDialog. """,
    demo_format="<|start_header_id|>user<|end_header_id|>\n\n{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{target_prefix}{target}<|eot_id|>",
    model_input_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{system_prompt}<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
)

add_to_catalog(
    format, "formats.llama3_multi_turn_chat_with_system_prompt", overwrite=True
)
