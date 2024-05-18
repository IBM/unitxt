from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import ListFieldValues, RenameFields
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="piqa"),
    preprocess_steps=[
        ListFieldValues(fields=["sol1", "sol2"], to_field="choices"),
        RenameFields(
            field_to_field={"goal": "question", "label": "answer"},
        ),
    ],
    task="tasks.qa.multiple_choice.open",
    templates="templates.qa.multiple_choice.open.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:question-answering",
            "task_ids:multiple-choice-qa",
            "annotations_creators:crowdsourced",
            "language_creators:crowdsourced",
            "language_creators:found",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:original",
            "language:en",
            "license:unknown",
            "croissant",
            "arxiv:1911.11641",
            "arxiv:1907.10641",
            "arxiv:1904.09728",
            "arxiv:1808.05326",
            "region:us",
        ]
    },
)
test_card(card, strict=False)
add_to_catalog(card, "cards.piqa", overwrite=True)
