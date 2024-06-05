from unitxt.blocks import (
    AddFields,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    SplitRandomMix,
    Task,
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
    task=Task(
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
        "The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card)
add_to_catalog(card, "cards.mrpc", overwrite=True)
