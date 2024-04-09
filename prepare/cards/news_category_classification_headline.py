import sys

from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.loaders import LoadFromKaggle
from unitxt.operators import Shuffle
from unitxt.test_utils.card import test_card

dataset_name = "news_category_classification_headline"


classlabels = [
    "ARTS",
    "ARTS & CULTURE",
    "BLACK VOICES",
    "BUSINESS",
    "COLLEGE",
    "COMEDY",
    "CRIME",
    "CULTURE & ARTS",
    "DIVORCE",
    "EDUCATION",
    "ENTERTAINMENT",
    "ENVIRONMENT",
    "FIFTY",
    "FOOD & DRINK",
    "GOOD NEWS",
    "GREEN",
    "HEALTHY LIVING",
    "HOME & LIVING",
    "IMPACT",
    "LATINO VOICES",
    "MEDIA",
    "MONEY",
    "PARENTING",
    "PARENTS",
    "POLITICS",
    "QUEER VOICES",
    "RELIGION",
    "SCIENCE",
    "SPORTS",
    "STYLE",
    "STYLE & BEAUTY",
    "TASTE",
    "TECH",
    "THE WORLDPOST",
    "TRAVEL",
    "U.S. NEWS",
    "WEDDINGS",
    "WEIRD NEWS",
    "WELLNESS",
    "WOMEN",
    "WORLD NEWS",
    "WORLDPOST",
]

card = TaskCard(
    loader=LoadFromKaggle(
        url="https://www.kaggle.com/datasets/rmisra/news-category-dataset"
    ),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        SplitRandomMix(
            {"train": "train[70%]", "validation": "train[10%]", "test": "train[20%]"}
        ),
        RenameFields(field_to_field={"headline": "text"}),
        RenameFields(field_to_field={"category": "label"}),
        AddFields(
            fields={
                "classes": classlabels,
                "text_type": "sentence",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
