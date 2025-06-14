from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    MapInstanceValues,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="fancyzhx/ag_news"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[87.5%]", "validation": "train[12.5%]", "test": "test"}
        ),
        MapInstanceValues(
            mappers={
                "label": {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}
            }
        ),
        Set(
            fields={
                "classes": ["World", "Sports", "Business", "Sci/Tech"],
                "text_type": "sentence",
            }
        ),
    ],
    task="tasks.classification.multi_class.topic_classification",
    templates="templates.classification.multi_class.all",
    __tags__={
        "annotations_creators": "found",
        "language": "en",
        "language_creators": "found",
        "license": "unknown",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "100K<n<1M",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": "topic-classification",
    },
    __description__=(
        "AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July 2004. The dataset is provided by the academic community for research purposes in data mining (clustering, classification, etc), information retrieval (ranking, search, etc)… See the full description on the dataset page: https://huggingface.co/datasets/fancyzhx/ag_news."
    ),
)
test_card(card, debug=False)
add_to_catalog(card, "cards.ag_news", overwrite=True)
