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
                "dataset_info_tags": [
                    "task_categories:text-generation",
                    "language_creators:crowdsourced",
                    "language_creators:expert-generated",
                    "language_creators:machine-generated",
                    "multilinguality:multilingual",
                    "size_categories:10K<n<100K",
                    "source_datasets:original",
                    "source_datasets:extended",
                    "language:afr",
                    "language:sqi",
                    "language:amh",
                    "language:ara",
                    "language:aze",
                    "language:bel",
                    "language:ben",
                    "language:bul",
                    "language:cat",
                    "language:ceb",
                    "language:ces",
                    "language:kur",
                    "language:cym",
                    "language:dan",
                    "language:deu",
                    "language:ell",
                    "language:eng",
                    "language:epo",
                    "language:est",
                    "language:eus",
                    "language:fin",
                    "language:fra",
                    "language:gla",
                    "language:gle",
                    "language:glg",
                    "language:guj",
                    "language:hat",
                    "language:hau",
                    "language:heb",
                    "language:hin",
                    "language:hun",
                    "language:hye",
                    "language:ibo",
                    "language:ind",
                    "language:isl",
                    "language:ita",
                    "language:jav",
                    "language:jpn",
                    "language:kan",
                    "language:kat",
                    "language:kaz",
                    "language:mon",
                    "language:khm",
                    "language:kir",
                    "language:kor",
                    "language:lao",
                    "language:lit",
                    "language:ltz",
                    "language:lav",
                    "language:mal",
                    "language:mar",
                    "language:mkd",
                    "language:mlt",
                    "language:mri",
                    "language:mya",
                    "language:nld",
                    "language:nor",
                    "language:nep",
                    "language:sot",
                    "language:pus",
                    "language:pes",
                    "language:mlg",
                    "language:pol",
                    "language:por",
                    "language:ron",
                    "language:rus",
                    "language:sin",
                    "language:slk",
                    "language:slv",
                    "language:smo",
                    "language:sna",
                    "language:snd",
                    "language:som",
                    "language:spa",
                    "language:srp",
                    "language:sun",
                    "language:swe",
                    "language:swa",
                    "language:tam",
                    "language:tel",
                    "language:tgk",
                    "language:tha",
                    "language:tur",
                    "language:ukr",
                    "language:urd",
                    "language:uzb",
                    "language:vie",
                    "language:xho",
                    "language:yid",
                    "language:yor",
                    "language:zho",
                    "language:msa",
                    "language:zul",
                    "language:ace",
                    "language:bjn",
                    "language:kas",
                    "language:kau",
                    "language:min",
                    "language:mni",
                    "language:taq",
                    "language:nso",
                    "license:apache-2.0",
                    "croissant",
                    "arxiv:2402.06619",
                    "region:us",
                ]
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
