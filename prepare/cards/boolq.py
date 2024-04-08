from unitxt.blocks import (
    AddFields,
    LoadHF,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    CastFields,
    MapInstanceValues,
    RenameFields,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="google/boolq"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields(
            {
                "text_a_type": "passage",
                "text_b_type": "question",
                "classes": ["yes", "no"],
                "type_of_relation": "answer",
            },
        ),
        CastFields(fields={"answer": "str"}),
        MapInstanceValues(mappers={"answer": {"True": "yes", "False": "no"}}),
        RenameFields(
            field_to_field={
                "passage": "text_a",
                "question": "text_b",
                "answer": "label",
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.all",
)

test_card(card, demos_taken_from="test")
add_to_catalog(card, "cards.boolq.classification", overwrite=True)

card = TaskCard(
    loader=LoadHF(path="google/boolq"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields(
            {
                "context_type": "passage",
                "choices": ["yes", "no"],
            },
        ),
        CastFields(fields={"answer": "str"}),
        MapInstanceValues(mappers={"answer": {"True": "yes", "False": "no"}}),
        RenameFields(
            field_to_field={
                "passage": "context",
            }
        ),
    ],
    task="tasks.qa.multiple_choice.with_context",
    templates="templates.qa.multiple_choice.with_context.all",
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(card, "cards.boolq.multiple_choice", overwrite=True)
