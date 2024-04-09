from unitxt.blocks import AddFields, LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Dictify, DuplicateBySubLists, Get, Wrap
from unitxt.dialog_operators import SerializeDialog
from unitxt.operators import CopyFields, ZipFieldValues
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="stanfordnlp/coqa"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields(fields={"context_type": "story"}),
        ZipFieldValues(
            fields=["questions", "answers/input_text"],
            to_field="dialog",
        ),
        Dictify(field="dialog", with_keys=["user", "system"], process_every_value=True),
        DuplicateBySubLists(field="dialog"),
        Get(field="dialog", item=-1, to_field="last_turn"),
        CopyFields(
            field_to_field={"last_turn/user": "question", "last_turn/system": "answer"},
        ),
        Wrap(
            field="answer",
            inside="list",
            to_field="answers",
        ),
        SerializeDialog(
            field="dialog",
            to_field="context",
            context_field="story",
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
        ZipFieldValues(
            fields=["questions", "answers/input_text"],
            to_field="dialog",
        ),
        Dictify(field="dialog", with_keys=["user", "system"], process_every_value=True),
        DuplicateBySubLists(field="dialog"),
        SerializeDialog(
            field="dialog",
            to_field="context",
            context_field="story",
            last_response_to_field="completion",
        ),
    ],
    task="tasks.completion.abstractive",
    templates="templates.completion.abstractive.all",
)

test_card(card)
add_to_catalog(card, "cards.coqa.completion", overwrite=True)
