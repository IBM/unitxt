from src.unitxt.blocks import AddFields, LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.dialog_operators import DialogSerializer
from src.unitxt.list_operators import ListsToListOfDicts, WrapWithList
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="stanfordnlp/coqa"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields(fields={"context_type": "story"}),
        ListsToListOfDicts(
            fields=["questions", "answers/input_text"],
            with_keys=["user", "system"],
            to_field="dialog",
            use_query=True,
        ),
        DialogSerializer(
            field="dialog",
            to_field="context",
            context_field="story",
            last_user_turn_to_field="question",
            last_system_turn_to_field="answer",
        ),
        WrapWithList(
            field="answer",
            to_field="answers",
        ),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
)

test_card(card)
add_to_catalog(card, "cards.coqa.qa", overwrite=True)

card = TaskCard(
    loader=LoadHF(path="stanfordnlp/coqa"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields(fields={"context_type": "dialog", "completion_type": "response"}),
        ListsToListOfDicts(
            fields=["questions", "answers/input_text"],
            with_keys=["user", "system"],
            to_field="dialog",
            use_query=True,
        ),
        DialogSerializer(
            field="dialog",
            to_field="context",
            context_field="story",
            last_system_turn_to_field="completion",
        ),
    ],
    task="tasks.completion.abstractive",
    templates="templates.completion.abstractive.all",
)

test_card(card)
add_to_catalog(card, "cards.coqa.completion", overwrite=True)
