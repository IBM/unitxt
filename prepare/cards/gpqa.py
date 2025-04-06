import unitxt
from unitxt.blocks import LoadHF
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    ListFieldValues,
    Rename,
    Set,
    ShuffleFieldValues,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

with unitxt.settings.context(allow_unverified_code=True):
    for subset in ["main", "diamond", "experts", "extended"]:
        card = TaskCard(
            loader=LoadHF(
                path="Idavidrein/gpqa",
                name="gpqa_" + subset,
                data_classification_policy=["public"],
            ),
            preprocess_steps=[
                RenameSplits({"train": "test"}),
                ListFieldValues(
                    fields=[
                        "Correct Answer",
                        "Incorrect Answer 1",
                        "Incorrect Answer 2",
                        "Incorrect Answer 3",
                    ],
                    to_field="choices",
                ),
                ShuffleFieldValues(field="choices"),
                Rename(field="Correct Answer", to_field="answer"),
                Rename(field="Subdomain", to_field="topic"),
                Rename(field="Question", to_field="question"),
                Set({"context_type": "situation"}),
            ],
            task="tasks.qa.multiple_choice.with_topic",
            templates="templates.qa.multiple_choice.with_topic.all",
            __description__=(
                """GPQA is a multiple-choice, Q&A dataset of very hard questions written and validated by experts in biology, physics, and chemistry. When attempting questions out of their own domain (e.g., a physicist answers a chemistry question), these experts get only 34 percent accuracy, despite spending >30m with full access to Google."""
            ),
            __tags__={
                "annotations_creators": "expert-generated",
                "arxiv": "2311.12022",
                "flags": ["NLU", "natural language understanding"],
                "language": "en",
                "language_creators": "other",
                "license": "cc-by-4.0",
                "multilinguality": "monolingual",
                "region": "us",
                "size_categories": "n<1K",
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

        if subset == "main":
            test_card(card, strict=False)
        add_to_catalog(card, "cards.gpqa." + subset, overwrite=True)
