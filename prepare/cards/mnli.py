from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="mnli"),
    preprocess_steps=[
        RenameSplits({"validation_matched": "validation"}),
        "splitters.small_no_test",
        RenameFields(field_to_field={"premise": "text_a", "hypothesis": "text_b"}),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}
        ),
        AddFields(
            fields={
                "type_of_relation": "entailment",
                "text_a_type": "premise",
                "text_b_type": "hypothesis",
                "classes": ["entailment", "neutral", "contradiction"],
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.all",
)

test_card(card)
add_to_catalog(card, "cards.mnli", overwrite=True)
