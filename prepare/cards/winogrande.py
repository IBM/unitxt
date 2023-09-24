from src.unitxt.blocks import LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import AddConstant, CastFields, ListFieldValues, RenameFields
from src.unitxt.test_utils.card import test_card

for subtask in ["debiased", "l", "m", "s", "xl", "xs"]:
    card = TaskCard(
        loader=LoadHF(path="winogrande", name=f"winogrande_{subtask}"),
        preprocess_steps=[
            "splitters.small_no_test",
            ListFieldValues(fields=["option1", "option2"], to_field="choices"),
            CastFields(fields={"answer": "int"}),
            AddConstant(field="answer", add=-1),
            RenameFields(
                field_to_field={"sentence": "question"},
            ),
        ],
        task="tasks.qa.multiple_choice.original",
        templates="templates.qa.multiple_choice.all",
    )
    test_card(card, tested_split="test")
    add_to_catalog(card, f"cards.winogrande.{subtask.replace('-', '_')}", overwrite=True)
