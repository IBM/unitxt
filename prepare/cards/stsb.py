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

test_card(card, strict=False)
add_to_catalog(card, "cards.stsb", overwrite=True)
