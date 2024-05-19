from unitxt.blocks import AddFields, CopyFields, LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="squad"),
    preprocess_steps=[
        "splitters.small_no_test",
        CopyFields(field_to_field=[["answers/text", "answers"]]),
        AddFields({"context_type": "passage"}),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": "1606.05250",
        "croissant": True,
        "language": "en",
        "language_creators": ["crowdsourced", "found"],
        "license": "cc-by-sa-4.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "extended|wikipedia",
        "task_categories": "question-answering",
        "task_ids": "extractive-qa",
    },
)

test_card(card)
add_to_catalog(card, "cards.squad", overwrite=True)
