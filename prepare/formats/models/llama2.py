from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

# see: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
# <s>[INST] <<SYS>>
# {{ system_prompt }}
# <</SYS>>

# {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
format = SystemFormat(
    demo_format="{source} [/INST] {target_prefix}{target} </s><s>[INST] ",
    model_input_format="[INST] <<SYS>>\n{system_prompt}\\N{instruction}<</SYS>>\n\n\n{demos}{source} [/INST] {target_prefix}",
)

add_to_catalog(format, "formats.llama2", overwrite=True)
