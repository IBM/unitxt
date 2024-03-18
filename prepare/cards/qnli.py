from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import RenameFields
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="qnli"),
    preprocess_steps=[
        "splitters.large_no_test",
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "classes": ["entailment", "not entailment"],
                "type_of_relation": "entailment",
                "text_a_type": "question",
                "text_b_type": "sentence",
            }
        ),
        RenameFields(
            field_to_field={
                "question": "text_a",
                "sentence": "text_b",
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.all",
)

test_card(card)
add_to_catalog(card, "cards.qnli", overwrite=True)
