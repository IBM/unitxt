import datasets
from unitxt.blocks import LoadHF, RenameFields, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import FilterByExpression
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card


def main():
    ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
    topics = list(set(ds["test"]["category"]))

    for topic in topics:
        card = TaskCard(
            loader=LoadHF(path="TIGER-Lab/MMLU-Pro"),
            preprocess_steps=[
                FilterByExpression(f"category == '{topic}'"),
                RenameSplits({"validation": "train"}),
                RenameFields(
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
                loader_limit=2 * len(ds["test"]),
            )
        add_to_catalog(card, f"cards.mmlu_pro.{topic.replace(' ','_')}", overwrite=True)


main()
