from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="stsb"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
        ),
        RenameFields(
            field_to_field={
                "sentence1": "text1",
                "sentence2": "text2",
                "label": "value",
            }
        ),
        AddFields(
            fields={"type_of_value": "similarity", "min_value": "1", "max_value": "5"}
        ),
    ],
    task="tasks.regression.bounded.pair",
    templates="templates.regression.bounded.pair.all",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.stsb", overwrite=True)
