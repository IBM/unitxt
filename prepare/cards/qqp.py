from unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    TaskCard,
    TemplatesList,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="qqp"),
    preprocess_steps=[
        "splitters.large_no_test",
        MapInstanceValues(
            mappers={"label": {"0": "not duplicated", "1": "duplicated"}}
        ),
        AddFields(
            fields={
                "choices": ["not duplicated", "duplicated"],
            }
        ),
    ],
    task=FormTask(
        inputs=["choices", "question1", "question2"],
        outputs=["label"],
        metrics=["metrics.accuracy"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given this question: {question1}, classify if this question: {question2} is {choices}.
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),
    __tags__={
        "annotations_creators": "other",
        "arxiv": "1804.07461",
        "coreference-nli": True,
        "croissant": True,
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
        "paraphrase-identification": True,
        "qa-nli": True,
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": [
            "acceptability-classification",
            "natural-language-inference",
            "semantic-similarity-scoring",
            "sentiment-classification",
            "text-scoring",
        ],
    },
)

test_card(card)
add_to_catalog(card, "cards.qqp", overwrite=True)
