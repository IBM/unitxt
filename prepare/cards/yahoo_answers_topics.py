import sys

from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import JoinStr, ListFieldValues, Shuffle
from unitxt.test_utils.card import test_card

dataset_name = "yahoo_answers_topics"

classes = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government",
]

mappers = {str(i): cls for i, cls in enumerate(classes)}

card = TaskCard(
    loader=LoadHF(path=f"{dataset_name}"),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        SplitRandomMix(
            {"train": "train[87.5%]", "validation": "train[12.5%]", "test": "test"}
        ),
        RenameFields(field_to_field={"topic": "label"}),
        MapInstanceValues(mappers={"label": mappers}),
        ListFieldValues(
            fields=["question_title", "question_content", "best_answer"],
            to_field="text",
        ),
        JoinStr(separator=" ", field="text", to_field="text"),
        AddFields(
            fields={
                "classes": classes,
                "text_type": "text",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
        "annotations_creators": "found",
        "croissant": True,
        "language": "en",
        "language_creators": "found",
        "license": "unknown",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "1M<n<10M",
        "source_datasets": "extended|other-yahoo-answers-corpus",
        "task_categories": "text-classification",
        "task_ids": "topic-classification",
    },
    __description__=(
        "Yahoo! Answers Topic Classification is text classification dataset. The dataset is the Yahoo! Answers corpus as of 10/25/2007. The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories. From all the answers and other meta-information, this dataset only used the best answer content and the main category information."
    ),
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
