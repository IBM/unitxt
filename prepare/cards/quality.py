import unitxt
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Explode
from unitxt.loaders import LoadCSV
from unitxt.operators import (
    Copy,
    MapInstanceValues,
    Set,
)
from unitxt.splitters import SplitRandomMix
from unitxt.test_utils.card import test_card

file_path = "https://raw.githubusercontent.com/nyu-mll/quality/05e85750d4c5444d2a0a4ad299f6df5f4df06068/data/v1.0.1/QuALITY.v1.0.1.htmlstripped."

with unitxt.settings.context(allow_unverified_code=True):
    card = TaskCard(
        loader=LoadCSV(
            files={"train": file_path + "train", "validation": file_path + "dev"},
            file_type="json",
            lines=True,
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            SplitRandomMix(
                {
                    "train": "train[80%]",
                    "validation": "train[20%]",
                    "test": "validation",
                }
            ),
            Copy(field="article", to_field="context"),
            Explode(field="questions", to_field="data"),
            Copy(field="data/question", to_field="question"),
            Copy(field="data/options", to_field="choices"),
            Copy(field="data/gold_label", to_field="answer"),
            MapInstanceValues(
                mappers={
                    "answer": {
                        "1": 0,
                        "2": 1,
                        "3": 2,
                        "4": 3,
                        "5": 4,
                    }
                }
            ),
            Set(fields={"context_type": "document"}),
        ],
        task="tasks.qa.multiple_choice.with_context",
        templates="templates.qa.multiple_choice.with_context.all",
        __description__=(
            """QuALITY (Question Answering with Long Input Texts, Yes!) is a multiple-choice reading comprehension dataset with long documents. The dataset comprises of documents from Project Gutenberg and questions written by human annotators. Each question has 4-5 answer choices, and requires understanding of the entire document to answer correctly. Questions are designed to test comprehensive understanding of the entire document, with various difficulty levels."""
        ),
        __tags__={
            "annotations_creators": "expert-generated",
            "language": ["en"],
            "license": "cc-by-4.0",
            "size_categories": ["10K<n<100K"],
            "task_categories": [
                "question-answering",
                "multiple-choice",
                "reading-comprehension",
            ],
            "multilinguality": "monolingual",
            "task_ids": ["extractive-qa", "reading-comprehension"],
        },
    )

    # Test and add the card to the catalog
    test_card(card, strict=False)
    add_to_catalog(card, "cards.quality", overwrite=True)
