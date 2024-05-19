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
        "annotations_creators": "crowdsourced",
        "arxiv": ["1911.11641", "1907.10641", "1904.09728", "1808.05326"],
        "croissant": True,
        "language": "en",
        "language_creators": ["crowdsourced", "found"],
        "license": "unknown",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "question-answering",
        "task_ids": "multiple-choice-qa",
    },
)
test_card(card, strict=False)
add_to_catalog(card, "cards.piqa", overwrite=True)
