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


add_to_catalog(
    SystemFormat(
        demo_format="{source}\n\n{target_prefix}{target}\n\n",
        model_input_format=(
            "[INST] " "{instruction}" "{demos}" "{source} [/INST]" "{target_prefix}"
        ),
    ),
    "formats.models.mistral.instruction.all_demos_in_one_turn",
    overwrite=True,
)
