from unitxt.blocks import (
    LoadHF,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="EdinburghNLP/xsum"),
    task="tasks.summarization.abstractive",
    preprocess_steps=[Wrap(field="summary", inside="list", to_field="summaries")],
    templates="templates.summarization.abstractive.all",
    __tags__={
        "annotations_creators": "found",
        "arxiv": "1808.08745",
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
        "Extreme Summarization (XSum) Dataset. There are three features:\n"
        "- document: Input news article. \n"
        "- summary: One sentence summary of the article. \n"
        "- id: BBC ID of the article… See the full description on the dataset page: https://huggingface.co/datasets/EdinburghNLP/xsum"
    ),
)

test_card(card)
add_to_catalog(card, "cards.xsum", overwrite=True)
