from src.unitxt.blocks import AddFields, LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import CastFields, RenameFields
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="hellaswag"),
    preprocess_steps=[
        "splitters.large_no_test",
        RenameFields(field_to_field={"ctx": "context", "activity_label": "topic", "endings": "choices"}),
        AddFields({"question": "Pick the best ending to the context."}),  # https://rowanzellers.com/hellaswag/
        RenameFields(field_to_field={"label": "answer"}),
        CastFields(fields={"answer": "int"}),
    ],
    task="tasks.qa.multiple_choice.contextual_with_topic",
    templates="templates.qa.multiple_choice.contextual_with_topic.all",
)
test_card(card)
add_to_catalog(card, f"cards.hellaswag", overwrite=True)
