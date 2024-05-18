from unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

default_splitter = SplitRandomMix(
    {"train": "train", "validation": "validation", "test": "test"}
)
add_to_catalog(default_splitter, "splitters.default", overwrite=True)


card = TaskCard(
    loader=LoadHF(path="glue", name="mrpc", streaming=False),
    preprocess_steps=[
        "splitters.default",
        MapInstanceValues(
            mappers={"label": {"0": "not equivalent", "1": "equivalent"}}
        ),
        AddFields(
            fields={
                "choices": ["not equivalent", "equivalent"],
            }
        ),
    ],
    task=FormTask(
        inputs=["choices", "sentence1", "sentence2"],
        outputs=["label"],
        metrics=["metrics.accuracy"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),
    __tags__={
        "dataset_info_tags": [
            "task_categories:text-classification",
            "task_ids:acceptability-classification",
            "task_ids:natural-language-inference",
            "task_ids:semantic-similarity-scoring",
            "task_ids:sentiment-classification",
            "task_ids:text-scoring",
            "annotations_creators:other",
            "language_creators:other",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:original",
            "language:en",
            "license:other",
            "qa-nli",
            "coreference-nli",
            "paraphrase-identification",
            "croissant",
            "arxiv:1804.07461",
            "region:us",
        ]
    },
)

test_card(card)
add_to_catalog(card, "cards.mrpc", overwrite=True)
