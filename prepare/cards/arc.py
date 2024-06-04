from unitxt.blocks import AddFields, LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import CopyFields, IndexOf, RenameFields
from unitxt.test_utils.card import test_card

subtasks = ["ARC-Challenge", "ARC-Easy"]

for subtask in subtasks:
    card = TaskCard(
        loader=LoadHF(path="ai2_arc", name=subtask),
        preprocess_steps=[
            AddFields({"topic": "science"}),
            RenameFields(field_to_field={"answerKey": "label", "choices": "_choices"}),
            CopyFields(
                field_to_field={"_choices/text": "choices", "_choices/label": "labels"}
            ),
            IndexOf(search_in="labels", index_of="label", to_field="answer"),
        ],
        task="tasks.qa.multiple_choice.with_topic",
        templates="templates.qa.multiple_choice.with_topic.all",
        __tags__={
            "annotations_creators": "found",
            "arxiv": "1803.05457",
            "flags": ["croissant"],
            "language": "en",
            "language_creators": "found",
            "license": "cc-by-sa-4.0",
            "multilinguality": "monolingual",
            "region": "us",
            "size_categories": "1K<n<10K",
            "source_datasets": "original",
            "task_categories": "question-answering",
            "task_ids": ["open-domain-qa", "multiple-choice-qa"],
        },
        __description__=(
            "A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. We are also including a corpus of over 14 million science sentences… See the full description on the dataset page: https://huggingface.co/datasets/allenai/ai2_arc."
        ),
    )
    test_card(card, strict=False)
    add_to_catalog(
        card, f'cards.ai2_arc.{subtask.replace("-", "_").lower()}', overwrite=True
    )
