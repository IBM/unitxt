from unitxt.blocks import (
    AddFields,
    ListToKeyValPairs,
    LoadHF,
    RenameFields,
    SerializeKeyValPairs,
    SplitRandomMix,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wiki_bio"),
    preprocess_steps=[
        SplitRandomMix({"train": "train", "validation": "val", "test": "test"}),
        ListToKeyValPairs(
            fields=["input_text/table/column_header", "input_text/table/content"],
            to_field="kvpairs",
        ),
        SerializeKeyValPairs(field_to_field=[["kvpairs", "input"]]),
        RenameFields(field_to_field={"target_text": "output"}),
        AddFields(
            fields={"type_of_input": "Key-Value pairs", "type_of_output": "Text"}
        ),
    ],
    task="tasks.generation",
    templates="templates.generation.all",
)

test_card(card)
add_to_catalog(card, "cards.wiki_bio", overwrite=True)
