from src.unitxt.blocks import (
    LoadHF,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.logging_utils import get_logger
from src.unitxt.operators import ListFieldValues
from src.unitxt.standard import StandardRecipeWithIndexes
from src.unitxt.test_utils.card import test_card

dataset_name = "CohereForAI/aya_evaluation_suite"
subsets = ["aya_human_annotated", "dolly_machine_translated", "dolly_human_edited"]
langs = ["eng", "fra", "deu", "spa", "por", "jpn"]
subset_to_langs = {
    "aya_human_annotated": langs,
    "dolly_machine_translated": langs,
    "dolly_human_edited": ["fra", "spa"],
}

for subset in subsets:
    for lang in subset_to_langs[subset]:
        card = TaskCard(
            loader=LoadHF(
                path=dataset_name,
                name=subset,
                streaming=True,
                filtering_lambda=f'lambda instance: instance["language"]=="{lang}"',
            ),
            preprocess_steps=[
                SplitRandomMix(
                    {"train": "test[90%]", "validation": "test[5%]", "test": "test[5%]"}
                ),
                RenameFields(
                    field_to_field={"inputs": "question", "targets": "answers"}
                ),
                ListFieldValues(fields=["answers"], to_field="answers"),
            ],
            task="tasks.qa.open",
            templates="templates.qa.open.all",
        )
        if lang == subset_to_langs[subset][0]:
            test_card(card, debug=False, strict=False)
        add_to_catalog(card, f"cards.cohere_for_ai.{subset}.{lang}", overwrite=True)

########################  to remove once done ############################
logger = get_logger()
recipe = StandardRecipeWithIndexes(
    template_card_index=1,
    card=f"cards.cohere_for_ai.{subsets[0]}.{langs[0]}",
    num_demos=2,
    demos_pool_size=10,
)
ms = recipe()
logger.info(ms)
train_as_list = list(ms["train"])
logger.info(len(train_as_list))
logger.info(train_as_list[0])
logger.info(train_as_list[1])
logger.info("+++++++++++1+++++++++++++++")
logger.info(train_as_list[0]["source"])
logger.info("+++++++++++2+++++++++++++++")
logger.info(train_as_list[1]["source"])
logger.info("+++++++++++3+++++++++++++++")
logger.info(train_as_list[2]["source"])
logger.info("+++++++++++4+++++++++++++++")
logger.info(train_as_list[3]["source"])
logger.info("+++++++++++done+++++++++++++++")
logger.info("done")
############# end of to remove once done ##################
