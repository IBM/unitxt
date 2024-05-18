from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    AddFields,
    IndexOf,
    ListFieldValues,
    RenameFields,
    ShuffleFieldValues,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="sciq"),
    preprocess_steps=[
        ListFieldValues(
            fields=["distractor1", "distractor2", "distractor3", "correct_answer"],
            to_field="choices",
        ),
        ShuffleFieldValues(field="choices"),
        IndexOf(search_in="choices", index_of="correct_answer", to_field="answer"),
        RenameFields(
            field_to_field={"support": "context"},
        ),
        AddFields({"context_type": "paragraph"}),
    ],
    task="tasks.qa.multiple_choice.with_context",
    templates="templates.qa.multiple_choice.with_context.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:question-answering",
            "task_ids:closed-domain-qa",
            "annotations_creators:no-annotation",
            "language_creators:crowdsourced",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:original",
            "language:en",
            "license:cc-by-nc-3.0",
            "croissant",
            "region:us",
        ]
    },
)
test_card(card, strict=False)
add_to_catalog(card, "cards.sciq", overwrite=True)
