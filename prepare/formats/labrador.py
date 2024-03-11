from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

format = SystemFormat(
    model_input_format="<|system|>\n{system_prompt}\n<|user|>\n{source}\n<|assistant|>\n{target_prefix}",
)

add_to_catalog(format, "formats.models.labrador.zero_shot", overwrite=True)

format = SystemFormat(
    format_args={"input_label": "Input:", "output_label": "Output:"},
    demo_format=(
        "{input_label}\n" "{source}\n" "{output_label}\n" "{target_prefix}{target}\n\n"
    ),
    model_input_format=(
        "<|system|>\n"
        "{system_prompt}\n"
        "<|user|>\n"
        "{instruction}\n"
        "Your response should only include the answer. Do not provide any further explanation.\n"
        "\n"
        "Here are some examples, complete the last one:\n"
        "{demos}"
        "{input_label}\n"
        "{source}\n"
        "{output_label}\n"
        "<|assistant|>\n"
        "{target_prefix}"
    ),
)

add_to_catalog(format, "formats.models.labrador.few_shot", overwrite=True)
