from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

# DeepSeek-Coder format and system prompt according to: https://github.com/deepseek-ai/deepseek-coder

format = SystemFormat(
    demo_format="### Instruction:\n{source}\n## Response:\n{target_prefix}{target}\n<EOT>\n",
    model_input_format="{system_prompt}\n{demos}### Instruction:\n{source}\n### Response:\n{target_prefix}",
)


add_to_catalog(format, "formats.deepseek_coder", overwrite=True)
