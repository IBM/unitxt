from src.unitxt.blocks import LoadHF
from src.unitxt.card import TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import ListFieldValues, MapInstanceValues, RenameFields
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="super_glue", name="copa"),
    preprocess_steps=[
        "splitters.small_no_test",
        ListFieldValues(fields=["choice1", "choice2"], to_field="choices"),
        RenameFields(field_to_field={"premise": "context", "label": "answer"}),
        MapInstanceValues(
            mappers={
                "question": {  # https://people.ict.usc.edu/~gordon/copa.html
                    "cause": "What was the cause of this?",
                    "effect": "What happened as a result?",
                }
            }
        ),
    ],
    task="tasks.qa.multiple_choice.contextual",
    templates="templates.qa.multiple_choice.context.all",
)

test_card(card)
add_to_catalog(card, "cards.copa", overwrite=True)
