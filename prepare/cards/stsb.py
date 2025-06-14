from unitxt.blocks import (
    LoadHF,
    Rename,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="nyu-mll/glue", name="stsb", splits=["train", "validation", "test"]
    ),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
        ),
        Rename(
            field_to_field={
                "sentence1": "text1",
                "sentence2": "text2",
                "label": "attribute_value",
            }
        ),
        Set(fields={"min_value": 1.0, "max_value": 5.0}),
    ],
    task="tasks.regression.two_texts.similarity",
    templates="templates.regression.two_texts.all",
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
        "The Semantic Textual Similarity Benchmark (Cer et al., 2017) is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data. Each pair is human-annotated with a similarity score from 1 to 5… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card, strict=False)
add_to_catalog(card, "cards.stsb", overwrite=True)
