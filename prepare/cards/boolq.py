from src.unitxt.blocks import AddFields, LoadHF, TaskCard, TemplatesList
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import CastFields
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="boolq"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields({"choices": ["False", "True"]}),
        CastFields(fields={"answer": "int"}),
    ],
    task="tasks.qa.multiple_choice.open",
    templates=TemplatesList(
        [
            "templates.qa.multiple_choice.lm_eval_harness",
        ]
    ),
)

test_card(card, tested_split="test")
add_to_catalog(card, "cards.boolq", overwrite=True)
