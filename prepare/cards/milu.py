from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Deduplicate,
    FilterByCondition,
    ListFieldValues,
    MapInstanceValues,
    Rename,
    Set,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

languages = [
    ["Bengali", "bn"],
    ["English", "en"],
    ["Gujarati", "gu"],
    ["Hindi", "hi"],
    ["Kannada", "kn"],
    ["Malayalam", "ml"],
    ["Marathi", "mr"],
    ["Odia", "or"],
    ["Punjabi", "pa"],
    ["Tamil", "ta"],
    ["Telugu", "te"],
]

subtasks = [
    "Arts & Humanities",
    "Business Studies",
    "Engineering & Tech",
    "Environmental Sciences",
    "Health & Medicine",
    "Law & Governance",
    "Science",
    "Social Sciences",
]

is_first = True
for language in languages:
    for subtask in subtasks:
        card = TaskCard(
            loader=LoadHF(
                path="ai4bharat/MILU",
                data_dir=language[0],
                splits=["validation", "test"],
            ),
            preprocess_steps=[
                FilterByCondition(values={"domain": subtask}, condition="eq"),
                Deduplicate(by=["question", "subject", "target"]),
                RenameSplits({"validation": "train"}),
                Rename(field_to_field={"target": "answer"}),
                MapInstanceValues(
                    mappers={
                        "answer": {
                            "option1": 0,
                            "option2": 1,
                            "option3": 2,
                            "option4": 3,
                        }
                    }
                ),
                ListFieldValues(
                    fields=["option1", "option2", "option3", "option4"],
                    to_field="choices",
                ),
                Set({"topic": subtask}),
            ],
            task="tasks.qa.multiple_choice.with_topic",
            templates=["templates.qa.multiple_choice.with_topic.all"],
            __tags__={
                "annotations_creators": "no-annotation",
                "arxiv": ["2411.02538"],
                "language": language[1],
                "language_creators": "expert-generated",
                "license": "CC BY 4.0",
                "multilinguality": "multilingual",
                "region": "in",
                "size_categories": "10K<n<100K",
                "source_datasets": "original",
                "task_categories": "question-answering",
                "task_ids": "multiple-choice-qa",
            },
            __description__=(
                "MILU (Multi-task Indic Language Understanding Benchmark) is a comprehensive evaluation dataset designed to assess the performance of Large Language Models (LLMs) across 11 Indic languages. It spans 8 domains and 41 subjects, reflecting both general and culturally specific knowledge from India. See the full description on the dataset page: https://huggingface.co/datasets/ai4bharat/MILU."
            ),
        )

        if is_first:
            test_card(card, strict=False)
            is_first = False

        subject = subtask.replace("&", "and").replace(" ", "_")
        add_to_catalog(
            card,
            f"cards.milu.{language[0]}.{subject}",
            overwrite=True,
        )
