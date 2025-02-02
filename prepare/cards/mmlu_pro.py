from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import Deduplicate, Rename, Set
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

topics = [
    "history",
    "philosophy",
    "business",
    "computer science",
    "physics",
    "biology",
    "engineering",
    "health",
    "psychology",
    "chemistry",
    "other",
    "economics",
    "math",
    "law",
]

for topic in topics:
    card = TaskCard(
        loader=LoadHF(
            path="TIGER-Lab/MMLU-Pro",
            filtering_lambda=f"lambda x: x['category'] == '{topic}'",
        ),
        preprocess_steps=[
            Deduplicate(by=["question", "options", "answer", "category"]),
            RenameSplits({"validation": "train"}),
            Rename(
                field_to_field={
                    "options": "choices",
                    "answer_index": "answer",
                }
            ),
            Set(
                fields={
                    "topic": topic,
                }
            ),
        ],
        task="tasks.qa.multiple_choice.with_topic",
        templates="templates.qa.multiple_choice.with_topic.all",
        __tags__={
            "annotations_creators": "no-annotation",
            "arxiv": ["2406.01574"],
            "language": "en",
            "language_creators": "expert-generated",
            "license": "mit",
            "multilinguality": "monolingual",
            "size_categories": "10K<n<100K",
            "source_datasets": "original",
            "task_categories": "question-answering",
            "task_ids": "multiple-choice-qa",
        },
        __description__=(
            "MMLU-Pro dataset is a more robust and challenging massive multi-task understanding dataset tailored to more rigorously benchmark large language models' capabilities. This dataset contains 12K complex questions across various disciplines."
        ),
    )
    if topic == topics[0]:
        test_card(
            card,
            strict=False,  # random generation here does not produce 0 results (MCQA)
            # loader_limit=2 * len(ds["test"]),
        )
    add_to_catalog(card, f"cards.mmlu_pro.{topic.replace(' ', '_')}", overwrite=True)
