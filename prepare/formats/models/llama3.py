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

# The following is a Llama-2 default prompt obtained from online sources
# See: https://developer.ibm.com/tutorials/awb-prompt-engineering-llama-2/
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


format = SystemFormat(
    demo_format="<|start_header_id|>user<|end_header_id|>\n\n"
    "{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{target_prefix}{target}<|eot_id|>",
    model_input_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    + DEFAULT_SYSTEM_PROMPT
    + "<|eot_id|>{demos}<|start_header_id|>user<|end_header_id|>\n\n"
    "{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_prefix}",
)

add_to_catalog(
    format,
    "formats.llama3_chat",
    overwrite=True,
)

format = SystemFormat(
    demo_format="<|start_header_id|>user<|end_header_id|>\n\n"
    "{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{target_prefix}{target}<|eot_id|>",
    model_input_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|>{demos}<|start_header_id|>user<|end_header_id|>\n\n"
    "{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_prefix}",
)

add_to_catalog(
    format,
    "formats.llama3_instruct",
    overwrite=True,
)
