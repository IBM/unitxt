from src.unitxt.blocks import LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import (
    AddFields,
    ExecuteExpression,
    ListFieldValues,
    RenameFields,
)
from src.unitxt.settings_utils import get_settings
from src.unitxt.test_utils.card import test_card

settings = get_settings()
orig_settings = settings.allow_unverified_code
settings.allow_unverified_code = True

card_abstractive = TaskCard(
    loader=LoadHF(path="multidoc2dial"),
    preprocess_steps=[
        RenameFields(
            field_to_field={"answers/text/0": "relevant_context"},
            use_query=True,
        ),
        ListFieldValues(fields=["utterance"], to_field="answer"),
        ExecuteExpression(expression="question.split('[SEP]')[0]", to_field="question"),
        AddFields({"context_type": "document"}),
    ],
    task="tasks.qa.with_context.abstractive",
    templates="templates.qa.with_context.all",
)

card_extractive = TaskCard(
    loader=LoadHF(path="multidoc2dial"),
    preprocess_steps=[
        RenameFields(
            field_to_field={"answers/text/0": "relevant_context"},
            use_query=True,
        ),
        ListFieldValues(fields=["relevant_context"], to_field="answer"),
        ExecuteExpression(expression="question.split('[SEP]')[0]", to_field="question"),
        AddFields({"context_type": "document"}),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
)

for name, card in zip(
    ["abstractive", "extractive"], [card_abstractive, card_extractive]
):
    test_card(card)
    add_to_catalog(card, f"cards.multidoc2dial.{name}", overwrite=True)

settings.allow_unverified_code = orig_settings
