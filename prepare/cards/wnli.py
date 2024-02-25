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
                "sentence1": "text_a",
                "sentence2": "text_b",
            }
        ),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "classes": ["entailment", "not entailment"],
                "type_of_relation": "entailment",
                "text_a_type": "premise",
                "text_b_type": "hypothesis",
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.all",
)

test_card(card)
add_to_catalog(card, "cards.wnli", overwrite=True)
