from src.unitxt.blocks import (
    AddFields,
    CopyFields,
    LoadHF,
    SerializeTableAsIndexedRowMajor,
    TaskCard,
    TruncateTableCells,
    TruncateTableRows,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wikitablequestions"),
    preprocess_steps=[
        "splitters.small_no_test",
        CopyFields(field_to_field=[["answers", "answer"]], use_query=True),
        AddFields({"context_type": "table"}),
        TruncateTableCells(max_length=15, table="table", text_output="answer"),
        TruncateTableRows(field="table", rows_to_keep=50),
        SerializeTableAsIndexedRowMajor(field_to_field=[["table", "context"]]),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
)

test_card(card)
add_to_catalog(card, "cards.wikitq", overwrite=True)
