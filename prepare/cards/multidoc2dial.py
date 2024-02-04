from src.unitxt.blocks import LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import ExecuteExpression, ListFieldValues, RenameFields
from src.unitxt.test_utils.card import test_card

card_abstractive = TaskCard(
    loader=LoadHF(path="multidoc2dial"),
    preprocess_steps=[
        RenameFields(
            field_to_field={"answers/text/0": "relevant_context"},
            use_query=True,
        ),
        ListFieldValues(fields=["utterance"], to_field="answer"),
        ExecuteExpression(expression="question.split('[SEP]')[0]", to_field="question"),
    ],
    task="tasks.qa.contextual.abstractive",
    templates="templates.qa.contextual.all",
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
    ],
    task="tasks.qa.contextual.extractive",
    templates="templates.qa.contextual.all",
)

for name, card in zip(
    ["abstractive", "extractive"], [card_abstractive, card_extractive]
):
    test_card(card)
    add_to_catalog(card, f"cards.multidoc2dial.{name}", overwrite=True)
