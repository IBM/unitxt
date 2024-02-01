from src.unitxt.blocks import (
    CopyFields,
    IndexedRowMajorTableSerializer,
    LoadHF,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wikitablequestions"),
    preprocess_steps=[
        "splitters.small_no_test",
        CopyFields(field_to_field=[["answers", "answer"]], use_query=True),
        IndexedRowMajorTableSerializer(field_to_field=[["table", "context"]]),
    ],
    task="tasks.qa.contextual.extractive",
    templates="templates.qa.contextual.all",
)

test_card(card)
add_to_catalog(card, "cards.wikitq", overwrite=True)
