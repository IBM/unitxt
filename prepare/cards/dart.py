from unitxt.blocks import (
    AddFields,
    CopyFields,
    LoadHF,
    RenameFields,
    SerializeTriples,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="dart"),
    preprocess_steps=[
        "splitters.small_no_test",
        SerializeTriples(field_to_field=[["tripleset", "serialized_triples"]]),
        RenameFields(field_to_field={"serialized_triples": "input"}),
        CopyFields(
            field_to_field={"annotations/text/0": "output"},
        ),
        AddFields(fields={"type_of_input": "Triples", "type_of_output": "Text"}),
    ],
    task="tasks.generation",
    templates="templates.generation.all",
)

test_card(card)
add_to_catalog(card, "cards.dart", overwrite=True)
