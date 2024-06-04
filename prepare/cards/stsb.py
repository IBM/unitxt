from unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="stsb"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
        ),
        RenameFields(
            field_to_field={
                "sentence1": "text1",
                "sentence2": "text2",
                "label": "attribute_value",
            }
        ),
        AddFields(
            fields={"attribute_name": "similarity", "min_value": 1.0, "max_value": 5.0}
        ),
    ],
    task="tasks.regression.two_texts",
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
        "Dataset Card for GLUE Dataset Summary GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems. Supported Tasks and Leaderboards The leaderboard for the GLUE benchmark can be found at this address. It comprises the following tasks: ax A manually-curated evaluation dataset for fine-grained… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card, strict=False)
add_to_catalog(card, "cards.stsb", overwrite=True)
