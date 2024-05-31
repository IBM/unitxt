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
        "annotations_creators": "other",
        "language": "en",
        "language_creators": "other",
        "license": "mit",
        "multilinguality": "monolingual",
        "region": "us",
        "singletons": [
            "croissant",
            "finance",
            "hedgefunds",
            "markets",
            "quant",
            "stocks",
            "twitter",
            "wallstreet",
        ],
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": "multi-class-classification",
    },
    __description__=(
        "Dataset Description\n"
        "The Twitter Financial News dataset is an English-language dataset containing an annotated corpus of finance-related tweets. This dataset is used to classify finance-related tweets for their topic.\n"
        "The dataset holds 21,107 documents annotated with 20 labels:\n"
        "topics = {\n"
        '"LABEL_0": "Analyst Update",\n'
        '"LABEL_1": "Fed | Central Banks",\n'
        '"LABEL_2": "Company | Product News",\n'
        '"LABEL_3": "Treasuries | Corporate Debt",\n'
        '"LABEL_4": "Dividend"… See the full description on the dataset page: https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic.'
    ),
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
