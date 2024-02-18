from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.logging_utils import get_logger
from src.unitxt.operators import FilterByCondition
from src.unitxt.standard import StandardRecipeWithIndexes
from src.unitxt.task import FormTask
from src.unitxt.templates import InputOutputTemplate
from src.unitxt.test_utils.card import test_card

logger = get_logger()

dataset_name = "CohereForAI/aya_evaluation_suite"
subsets = ["aya_human_annotated", "dolly_machine_translated"]  # , "dolly_human_edited"]
langs = ["eng", "fra", "deu", "spa", "por", "jpn"]

for subset in subsets:
    for lang in langs:
        card = TaskCard(
            loader=LoadHF(path=dataset_name, name=subset, streaming=True),
            preprocess_steps=[
                SplitRandomMix(
                    {"train": "test[90%]", "validation": "test[5%]", "test": "test[5%]"}
                ),
                FilterByCondition(values={"language": lang}, condition="eq"),
            ],
            task=FormTask(
                inputs=["inputs"], outputs=["targets"], metrics=["metrics.rouge"]
            ),
            templates=[
                InputOutputTemplate(
                    input_format="Question: {inputs}",
                    output_format="{targets}",
                    instruction="Answer the following question.\n",
                    target_prefix="Answer: ",
                )
            ],
        )
        if lang == langs[0]:
            recipe = StandardRecipeWithIndexes(
                template_card_index=0, card=card, num_demos=2, demos_pool_size=20
            )
            ms = recipe()
            logger.info(ms)
            train_as_list = list(ms["train"])
            logger.info(len(train_as_list))
            logger.info(train_as_list[0])
            logger.info(train_as_list[0]["source"])
            logger.info("done")
            test_card(
                card, debug=False, loader_limit=25000
            )  # 25000 to reach every language
        add_to_catalog(card, f"cards.cohere_for_ai.{subset}.{lang}", overwrite=True)

# recipe = StandardRecipeWithIndexes(template_card_index=0, card=card)
# ms = recipe()
# print(ms)
# test_as_list = list(ms["test"])
# print(len(test_as_list))
# print(test_as_list[0])
# print(test_as_list[1])
# langs = test_as_list[0]['target'].split(", ")
# print(langs)
# print(len(langs))
# print(sorted(langs))
# print('done')
