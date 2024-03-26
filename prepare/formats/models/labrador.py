from unitxt import get_logger
from unitxt.standard import StandardRecipe
from unitxt.templates import InputOutputTemplate

from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat, WWoInstructionWWoFewShotDynamicFormat

zero_shot_with_instruction_format = SystemFormat(
    model_input_format=(
        "<|system|>\n{system_prompt}\n<|user|>"
        "{instruction}\n"
        "Your response should only include the answer. Do not provide any further explanation.\n"
        "\n{source}\n<|assistant|>\n{target_prefix}"
    ),
)

zero_shot_no_instruction_format = SystemFormat(
    model_input_format="<|system|>\n{system_prompt}\n<|user|>\n{source}\n<|assistant|>\n{target_prefix}",
)

few_shot_with_instruction_format = SystemFormat(
    demo_format="{source}\n{target_prefix}{target}\n\n",
    model_input_format=(
        "<|system|>\n"
        "{system_prompt}\n"
        "<|user|>\n"
        "{instruction}\n"
        "Your response should only include the answer. Do not provide any further explanation.\n"
        "\n"
        "Here are some examples, complete the last one:\n"
        "{demos}"
        "{source}\n"
        "{target_prefix}"
        "<|assistant|>\n"
    ),
)

few_shot_without_instruction_format = SystemFormat(
    demo_format="{source}\n{target_prefix}{target}\n\n",
    model_input_format=(
        "<|system|>\n"
        "{system_prompt}\n"
        "<|user|>\n"
        "Here are some examples, complete the last one:\n"
        "{demos}"
        "{source}\n"
        "{target_prefix}"
        "<|assistant|>\n"
    ),
)

format = WWoInstructionWWoFewShotDynamicFormat(
    few_shot_format=few_shot_without_instruction_format,
    few_shot_with_instruction_format=few_shot_with_instruction_format,
    zero_shot_format=zero_shot_no_instruction_format,
    zero_shot_with_instruction_format=zero_shot_with_instruction_format,
)

add_to_catalog(format, "formats.models.labrador", overwrite=True)


if __name__ == "__main__":
    logger = get_logger()
    for num_demos in [0, 3]:
        for instruction in ["", "THIS IS MY INSTRUCTION"]:
            recipe = StandardRecipe(
                card="cards.wnli",
                template=InputOutputTemplate(
                    input_format="{text_a}",
                    output_format="{label}",
                    instruction=instruction,
                ),
                format="formats.models.labrador",
                num_demos=num_demos,
                demos_pool_size=num_demos,
            )
            stream = recipe()
            logger.info(
                f'****** Source of a single example num demos={num_demos}, instruction={instruction}:\n{next(iter(stream["train"]))["source"]}\n\n'
            )
