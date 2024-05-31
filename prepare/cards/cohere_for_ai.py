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
                "singletons": ["croissant"],
                "size_categories": "10K<n<100K",
                "source_datasets": ["original", "extended"],
                "task_categories": "text-generation",
            },
            __description__=(
                "Dataset Summary\n"
                "Aya Evaluation Suite contains a total of 26,750 open-ended conversation-style prompts to evaluate multilingual open-ended generation quality.To strike a balance between language coverage and the quality that comes with human curation, we create an evaluation suite that includes:\n"
                "human-curated examples in 7 languages (tur, eng, yor, arb, zho, por, tel) → aya-human-annotated.\n"
                "machine-translations of handpicked examples into 101 languages →… See the full description on the dataset page: https://huggingface.co/datasets/CohereForAI/aya_evaluation_suite."
            ),
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
