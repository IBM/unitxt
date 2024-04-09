from unitxt.blocks import (
    AddFields,
    LoadHF,
    SerializeTableAsIndexedRowMajor,
    TaskCard,
    TruncateTableCells,
    TruncateTableRows,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wikitablequestions"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields({"context_type": "table"}),
        TruncateTableCells(max_length=15, table="table", text_output="answers"),
        TruncateTableRows(field="table", rows_to_keep=50),
        SerializeTableAsIndexedRowMajor(field_to_field=[["table", "context"]]),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
)

test_card(card)
add_to_catalog(card, "cards.wikitq", overwrite=True)
