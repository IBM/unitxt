from src.unitxt.blocks import LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import (
    IndexOf,
    ListFieldValues,
    RenameFields,
    ShuffleFieldValues,
)
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="sciq"),
    preprocess_steps=[
        ListFieldValues(
            fields=["distractor1", "distractor2", "distractor3", "correct_answer"],
            to_field="choices",
        ),
        ShuffleFieldValues(field="choices"),
        IndexOf(search_in="choices", index_of="correct_answer", to_field="answer"),
        RenameFields(
            field_to_field={"support": "context"},
        ),
    ],
    task="tasks.qa.multiple_choice.contextual",
    templates="templates.qa.multiple_choice.contextual.all",
)
test_card(card)
add_to_catalog(card, "cards.sciq", overwrite=True)
