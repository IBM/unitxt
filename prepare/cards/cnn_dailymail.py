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
        "annotations_creators": "no-annotation",
        "flags": ["croissant"],
        "language": "en",
        "language_creators": "found",
        "license": "apache-2.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "100K<n<1M",
        "source_datasets": "original",
        "task_categories": "summarization",
        "task_ids": "news-articles-summarization",
    },
    __description__=(
        "Dataset Card for CNN Dailymail Dataset\n"
        "Dataset Summary\n"
        "The CNN / DailyMail Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The current version supports both extractive and abstractive summarization, though the original version was created for machine reading and comprehension and abstractive question answering.\n"
        "Supported Tasks and Leaderboards… See the full description on the dataset page: https://huggingface.co/datasets/cnn_dailymail."
    ),
)

test_card(card)
add_to_catalog(card, "cards.cnn_dailymail", overwrite=True)
