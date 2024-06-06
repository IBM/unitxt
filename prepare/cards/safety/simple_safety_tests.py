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
    __description__="100 test prompts across five harm areas that LLMs, for the vast majority of applications, should refuse to comply with.",
    __tags__={
        "languages": ["english"],
        "urls": {"arxiv": "https://arxiv.org/abs/2311.08370"},
    },
)

test_card(
    card, format="formats.empty", strict=False, demos_taken_from="test", num_demos=0
)
add_to_catalog(card, "cards.safety.simple_safety_tests", overwrite=True)
