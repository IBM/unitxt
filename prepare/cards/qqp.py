from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    Set,
    Task,
    TaskCard,
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
        Set(
            fields={
                "choices": ["not duplicated", "duplicated"],
            }
        ),
    ],
    task=Task(
        input_fields=["choices", "question1", "question2"],
        reference_fields=["label"],
        metrics=["metrics.accuracy"],
    ),
    templates=[
        InputOutputTemplate(
            input_format="""
                    Given this question: {question1}, classify if this question: {question2} is {choices}.
                """.strip(),
            output_format="{label}",
        ),
    ],
    __tags__={
        "annotations_creators": "other",
        "arxiv": "1804.07461",
        "flags": ["coreference-nli", "paraphrase-identification", "qa-nli"],
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
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
    __description__=(
        "The Quora Question Pairs2 dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card)
add_to_catalog(card, "cards.qqp", overwrite=True)
