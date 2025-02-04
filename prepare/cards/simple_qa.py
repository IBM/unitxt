import unitxt
from unitxt.blocks import LoadHF
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.operators import (
    Rename,
)
from unitxt.test_utils.card import test_card

with unitxt.settings.context(allow_unverified_code=True):
    card = TaskCard(
        loader=LoadHF(path="basicv8vc/SimpleQA", data_classification_policy=["public"]),
        preprocess_steps=[
            Rename(field="problem", to_field="question"),
            Wrap(field="answer", inside="list", to_field="answers"),
        ],
        task="tasks.qa.open",
        templates="templates.qa.open.all",
        __description__=(
            """A factuality benchmark called SimpleQA that measures the ability for language models to answer short, fact-seeking questions."""
        ),
        __tags__={
            "annotations_creators": "expert-generated",
            "arxiv": "1904.09728",
            "flags": ["NLU", "natural language understanding"],
            "language": "en",
            "language_creators": "other",
            "license": "mit",
            "multilinguality": "monolingual",
            "region": "us",
            "size_categories": "10K<n<100K",
            "source_datasets": "extended|other",
            "task_categories": [
                "text-classification",
                "token-classification",
                "question-answering",
            ],
            "task_ids": [
                "natural-language-inference",
                "word-sense-disambiguation",
            ],
        },
    )

    test_card(card, strict=False)
    add_to_catalog(card, "cards.simple_qa", overwrite=True)
