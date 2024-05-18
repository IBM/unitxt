from unitxt.blocks import (
    AddFields,
    LoadHF,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="EdinburghNLP/xsum"),
    preprocess_steps=[
        AddFields(fields={"document_type": "document"}),
    ],
    task="tasks.summarization.abstractive",
    templates="templates.summarization.abstractive.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:summarization",
            "task_ids:news-articles-summarization",
            "annotations_creators:found",
            "language_creators:found",
            "multilinguality:monolingual",
            "size_categories:100K<n<1M",
            "source_datasets:original",
            "language:en",
            "license:unknown",
            "croissant",
            "arxiv:1808.08745",
            "region:us",
        ]
    },
)

test_card(card)
add_to_catalog(card, "cards.xsum", overwrite=True)
