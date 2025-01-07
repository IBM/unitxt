import unitxt
from unitxt.blocks import LoadHF
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.operators import (
    Rename,
    Set,
    WikipediaFetcher,
)
from unitxt.processors import LiteralEval
from unitxt.test_utils.card import test_card

with unitxt.settings.context(allow_unverified_code=True):
    card = TaskCard(
        loader=LoadHF(
            path="google/frames-benchmark", data_classification_policy=["public"]
        ),
        preprocess_steps=[
            Rename(field="Prompt", to_field="question"),
            Rename(field="Answer", to_field="answer"),
            Wrap(field="answer", inside="list", to_field="answers"),
            LiteralEval(field="wiki_links", to_field="context"),
            WikipediaFetcher(field="context", process_every_value=True),
            Set(fields={"context_type": "wikipedia articles"}),
        ],
        task="tasks.qa.with_context",
        templates="templates.qa.with_context.all",
        __description__=(
            """FRAMES is a comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning."""
        ),
        __tags__={
            "annotations_creators": "expert-generated",
            "arxiv": "1904.09728",
            "flags": ["NLU", "natural language understanding"],
            "language": "en",
            "language_creators": "other",
            "license": "other",
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
                "coreference-resolution",
                "extractive-qa",
            ],
        },
    )

    test_card(card, strict=False)
    add_to_catalog(card, "cards.frames", overwrite=True)
