from src.unitxt.blocks import AddFields, CopyFields, LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="squad"),
    preprocess_steps=[
        "splitters.small_no_test",
        CopyFields(field_to_field=[["answers/text", "answers"]]),
        AddFields({"context_type": "passage"}),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
)

test_card(card)
add_to_catalog(card, "cards.squad", overwrite=True)
