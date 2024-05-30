from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    TaskCard,
)
from unitxt.operators import (
    AddFields,
    RenameFields,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="Bertievidgen/SimpleSafetyTests", data_classification_policy=["public"]
    ),
    preprocess_steps=[
        RenameFields(field_to_field={"prompt": "input"}),
        AddFields(
            fields={
                "output": "None",
                "type_of_input": "question",
                "type_of_output": "answer",
                "prediction_type": "str",
            }
        ),
    ],
    task="tasks.generation",
    templates=["templates.generation.empty"],
    __tags__={},
)

test_card(
    card, format="formats.empty", strict=False, demos_taken_from="test", num_demos=0
)
add_to_catalog(card, "cards.safety.simple_safety_tests", overwrite=True)
