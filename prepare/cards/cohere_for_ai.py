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
            __tags__={
                "arxiv": "2402.06619",
                "croissant": True,
                "language": [
                    "afr",
                    "sqi",
                    "amh",
                    "ara",
                    "aze",
                    "bel",
                    "ben",
                    "bul",
                    "cat",
                    "ceb",
                    "ces",
                    "kur",
                    "cym",
                    "dan",
                    "deu",
                    "ell",
                    "eng",
                    "epo",
                    "est",
                    "eus",
                    "fin",
                    "fra",
                    "gla",
                    "gle",
                    "glg",
                    "guj",
                    "hat",
                    "hau",
                    "heb",
                    "hin",
                    "hun",
                    "hye",
                    "ibo",
                    "ind",
                    "isl",
                    "ita",
                    "jav",
                    "jpn",
                    "kan",
                    "kat",
                    "kaz",
                    "mon",
                    "khm",
                    "kir",
                    "kor",
                    "lao",
                    "lit",
                    "ltz",
                    "lav",
                    "mal",
                    "mar",
                    "mkd",
                    "mlt",
                    "mri",
                    "mya",
                    "nld",
                    "nor",
                    "nep",
                    "sot",
                    "pus",
                    "pes",
                    "mlg",
                    "pol",
                    "por",
                    "ron",
                    "rus",
                    "sin",
                    "slk",
                    "slv",
                    "smo",
                    "sna",
                    "snd",
                    "som",
                    "spa",
                    "srp",
                    "sun",
                    "swe",
                    "swa",
                    "tam",
                    "tel",
                    "tgk",
                    "tha",
                    "tur",
                    "ukr",
                    "urd",
                    "uzb",
                    "vie",
                    "xho",
                    "yid",
                    "yor",
                    "zho",
                    "msa",
                    "zul",
                    "ace",
                    "bjn",
                    "kas",
                    "kau",
                    "min",
                    "mni",
                    "taq",
                    "nso",
                ],
                "language_creators": [
                    "crowdsourced",
                    "expert-generated",
                    "machine-generated",
                ],
                "license": "apache-2.0",
                "multilinguality": "multilingual",
                "region": "us",
                "size_categories": "10K<n<100K",
                "source_datasets": ["original", "extended"],
                "task_categories": "text-generation",
            },
        )

        from copy import deepcopy

        card_for_test = deepcopy(card)
        card_for_test.task.metrics = [
            "metrics.rag.correctness.llama_index_by_mock",
        ]

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
