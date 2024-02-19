from src.unitxt.blocks import LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import CastFields, RenameFields
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="hellaswag"),
    preprocess_steps=[
        "splitters.large_no_test",
        RenameFields(
            field_to_field={
                "ctx": "context",
                "activity_label": "topic",
                "endings": "choices",
            }
        ),
        RenameFields(field_to_field={"label": "answer"}),
        CastFields(fields={"answer": "int"}),
    ],
    task="tasks.completion.multiple_choice.standard",
    templates="templates.completion.multiple_choice.all",
)
test_card(card, debug=False)
add_to_catalog(card, "cards.hellaswag", overwrite=True)
