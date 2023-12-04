from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="wnli"),
    preprocess_steps=[
        "splitters.small_no_test",
        RenameFields(
            field_to_field={
                "sentence1": "premise",
                "sentence2": "hypothesis",
            }
        ),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "choices": ["entailment", "not entailment"],
            }
        ),
    ],
    task="tasks.nli",
    templates="templates.classification.nli.all",
)

test_card(card)
add_to_catalog(card, "cards.wnli", overwrite=True)
