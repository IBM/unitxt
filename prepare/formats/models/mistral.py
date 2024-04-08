from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

add_to_catalog(
    SystemFormat(
        demo_format="{source} [/INST]{target_prefix}{target}</s> [INST] ",
        model_input_format=(
            "[INST] " "{instruction}" "{demos}" "{source} [/INST]" "{target_prefix}"
        ),
    ),
    "formats.models.mistral.instruction",
    overwrite=True,
)

add_to_catalog(
    SystemFormat(
        demo_format="{source} [/INST]{target_prefix}{target}</s> [INST] ",
        model_input_format=(
            "[INST] "
            "{system_prompt}\n{instruction}"
            "{demos}"
            "{source} [/INST]"
            "{target_prefix}"
        ),
    ),
    "formats.models.mistral.instruction.with_system_prompt",
    overwrite=True,
)
