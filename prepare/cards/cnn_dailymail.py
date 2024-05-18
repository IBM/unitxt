from unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="cnn_dailymail", name="3.0.0"),
    preprocess_steps=[
        RenameFields(field_to_field={"article": "document", "highlights": "summary"}),
        AddFields(fields={"document_type": "article"}),
    ],
    task="tasks.summarization.abstractive",
    templates="templates.summarization.abstractive.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:summarization",
            "task_ids:news-articles-summarization",
            "annotations_creators:no-annotation",
            "language_creators:found",
            "multilinguality:monolingual",
            "size_categories:100K<n<1M",
            "source_datasets:original",
            "language:en",
            "license:apache-2.0",
            "croissant",
            "region:us",
        ]
    },
)

test_card(card)
add_to_catalog(card, "cards.cnn_dailymail", overwrite=True)
