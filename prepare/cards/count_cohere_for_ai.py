from src.unitxt.blocks import (
    LoadHF,
    TaskCard,
)
from src.unitxt.logging_utils import get_logger
from src.unitxt.operators import ExtractFieldValues
from src.unitxt.standard import StandardRecipeWithIndexes
from src.unitxt.task import FormTask
from src.unitxt.templates import InputOutputTemplate

logger = get_logger()


dataset_name = "CohereForAI/aya_evaluation_suite"
# subset = "aya_human_annotated"
# subset = "dolly_machine_translated"
subset = "dolly_human_edited"

card = TaskCard(
    loader=LoadHF(path=dataset_name, name=subset, streaming=True),
    preprocess_steps=[
        ExtractFieldValues(field="language", to_field="all_langs", stream_name="test")
    ],
    task=FormTask(inputs=[], outputs=["all_langs"], metrics=[]),
    templates=[
        InputOutputTemplate(
            input_format="",
            output_format="{all_langs}",
        )
    ],
)

recipe = StandardRecipeWithIndexes(template_card_index=0, card=card)
ms = recipe()
logger.info(ms)
test_as_list = list(ms["test"])
logger.info(len(test_as_list))
logger.info(test_as_list[0])
logger.info(test_as_list[1])
langs = test_as_list[0]["target"].split(", ")
logger.info(langs)
logger.info(len(langs))
logger.info(sorted(langs))
logger.info("done")
