import sys

from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import Shuffle
from unitxt.test_utils.card import test_card

dataset_name = "dbpedia_14"

classes = [
    "Company",
    "Educational Institution",
    "Artist",
    "Athlete",
    "Office Holder",
    "Mean Of Transportation",
    "Building",
    "Natural Place",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Written Work",
]

mappers = {str(i): cls for i, cls in enumerate(classes)}

card = TaskCard(
    loader=LoadHF(path=f"{dataset_name}"),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        SplitRandomMix(
            {"train": "train[87.5%]", "validation": "train[12.5%]", "test": "test"}
        ),
        MapInstanceValues(mappers={"label": mappers}),
        RenameFields(field_to_field={"content": "text"}),
        AddFields(
            fields={
                "classes": classes,
                "text_type": "paragraph",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:text-classification",
            "task_ids:topic-classification",
            "annotations_creators:machine-generated",
            "language_creators:crowdsourced",
            "multilinguality:monolingual",
            "size_categories:100K<n<1M",
            "source_datasets:original",
            "language:en",
            "license:cc-by-sa-3.0",
            "croissant",
            "arxiv:1509.01626",
            "region:us",
        ]
    },
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
