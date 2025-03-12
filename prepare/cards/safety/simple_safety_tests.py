from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    Task,
    TaskCard,
)
from unitxt.operators import (
    Rename,
    Set,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="Bertievidgen/SimpleSafetyTests", data_classification_policy=["public"]
    ),
    preprocess_steps=[
        Rename(field_to_field={"prompt": "input"}),
        Set(
            fields={
                "output": "",
            }
        ),
    ],
    task=Task(
        input_fields={"input": str, "id": str, "harm_area": str, "category": str},
        reference_fields={"output": str},
        prediction_type=str,
        metrics=[
            "metrics.granite_guardian.assistant_risk.harm[prediction_type=str,user_message_field=input,assistant_message_field=output,score_prefix=graniteguardian_]",
            "metrics.llm_as_judge.safety.llamaguard[score_prefix=llamaguard_]"
        ],
    ),
    templates=["templates.generation.empty"],
    __description__="100 test prompts across five harm areas that LLMs, for the vast majority of applications, should refuse to comply with.",
    __tags__={
        "languages": ["english"],
        "urls": {"arxiv": "https://arxiv.org/abs/2311.08370"},
    },
)

test_card(card, strict=False, demos_taken_from="test", num_demos=0)

add_to_catalog(card, "cards.safety.simple_safety_tests", overwrite=True)
