import sys

from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import Shuffle
from unitxt.test_utils.card import test_card

dataset_name = "financial_tweets"

mappers = {
    "0": "analyst update",
    "1": "fed and central banks",
    "2": "company and product news",
    "3": "treasuries and corporate debt",
    "4": "dividend",
    "5": "earnings",
    "6": "energy and oil",
    "7": "financials",
    "8": "currencies",
    "9": "general News and opinion",
    "10": "gold, metals and materials",
    "11": "initial public offering",
    "12": "legal and regulation",
    "13": "mergers, acquisitions and investments",
    "14": "macro",
    "15": "markets",
    "16": "politics",
    "17": "personnel change",
    "18": "stock commentary",
    "19": "stock movement",
}


card = TaskCard(
    loader=LoadHF(path="zeroshot/twitter-financial-news-topic"),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        SplitRandomMix(
            {
                "train": "train[85%]",
                "validation": "train[15%]",
                "test": "validation",
            }  # TODO see the mapping due to sizes?
        ),
        MapInstanceValues(mappers={"label": mappers}),
        AddFields(
            fields={
                "classes": list(mappers.values()),
                "text_type": "tweet",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:text-classification",
            "task_ids:multi-class-classification",
            "annotations_creators:other",
            "language_creators:other",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:original",
            "language:en",
            "license:mit",
            "twitter",
            "finance",
            "markets",
            "stocks",
            "wallstreet",
            "quant",
            "hedgefunds",
            "croissant",
            "region:us",
        ]
    },
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
