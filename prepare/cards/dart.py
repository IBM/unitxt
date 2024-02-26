from src.unitxt.blocks import (
    AddFields,
    CopyFields,
    LoadHF,
    RenameFields,
    SerializeTriples,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="dart"),
    preprocess_steps=[
        "splitters.small_no_test",
        SerializeTriples(field_to_field=[["tripleset", "serialized_triples"]]),
        RenameFields(field_to_field={"serialized_triples": "input"}),
        CopyFields(
            field_to_field={"annotations/text/0": "output"},
            use_query=True,
        ),
        AddFields(fields={"type_of_input": "Triples"}),
    ],
    task="tasks.generation",
    templates="templates.generation.all",
)

test_card(card)
add_to_catalog(card, "cards.dart", overwrite=True)
