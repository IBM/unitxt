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
        "annotations_creators": "found",
        "arxiv": "1808.08745",
        "flags": ["croissant"],
        "language": "en",
        "language_creators": "found",
        "license": "unknown",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "100K<n<1M",
        "source_datasets": "original",
        "task_categories": "summarization",
        "task_ids": "news-articles-summarization",
    },
    __description__=(
        "Extreme Summarization (XSum) Dataset. There are three features: - document: Input news article. - summary: One sentence summary of the article. - id: BBC ID of the article.\n"
    ),
)

test_card(card)
add_to_catalog(card, "cards.xsum", overwrite=True)
