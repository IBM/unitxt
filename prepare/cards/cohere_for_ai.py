from unitxt.blocks import (
    LoadHF,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import ListFieldValues
from unitxt.test_utils.card import test_card

dataset_name = "CohereForAI/aya_evaluation_suite"
subsets = ["aya_human_annotated", "dolly_machine_translated", "dolly_human_edited"]
subset_to_langs = {
    "aya_human_annotated": ["eng", "zho", "arb", "yor", "tur", "tel", "por"],
    "dolly_machine_translated": ["eng", "fra", "deu", "spa", "por", "jpn"],
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
            task="tasks.qa.open[metrics=[metrics.rag.correctness.llama_index_by_gpt_3_5_turbo]]",
            templates="templates.qa.open.all",
        )

        from copy import deepcopy

        card_for_test = deepcopy(card)
        from unitxt.blocks import (
            FormTask,
        )

        card_for_test.task = FormTask(
            inputs=["question"],
            outputs=["answers"],
            metrics=["metrics.rag.correctness.llama_index_by_mock"],
        )

        test_card(card_for_test, debug=False, strict=False)
        add_to_catalog(
            card,
            f"cards.cohere_for_ai.{subset}.{lang}",
            overwrite=True,
            catalog_path="src/unitxt/catalog",
        )

########################  to remove once done ############################
# logger = get_logger()
# recipe = StandardRecipeWithIndexes(
#     template_card_index=1,
#     card=f"cards.cohere_for_ai.{subsets[0]}.{langs[0]}",
#     num_demos=2,
#     demos_pool_size=10,
# )
# ms = recipe()
# logger.info(ms)
# train_as_list = list(ms["train"])
# logger.info(len(train_as_list))
# logger.info(train_as_list[0])
# logger.info(train_as_list[1])
# logger.info("+++++++++++1+++++++++++++++")
# logger.info(train_as_list[0]["source"])
# logger.info("+++++++++++2+++++++++++++++")
# logger.info(train_as_list[1]["source"])
# logger.info("+++++++++++3+++++++++++++++")
# logger.info(train_as_list[2]["source"])
# logger.info("+++++++++++4+++++++++++++++")
# logger.info(train_as_list[3]["source"])
# logger.info("+++++++++++done+++++++++++++++")
# logger.info("done")
############# end of to remove once done ##################
