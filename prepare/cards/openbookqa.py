from src.unitxt.blocks import LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import IndexOf, RenameFields
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="openbookqa"),
    preprocess_steps=[
        RenameFields(
            field_to_field={"choices/text": "choices_text", "choices/label": "labels"},
            use_query=True,
        ),
        RenameFields(
            field_to_field={"choices_text": "choices", "question_stem": "question"},
        ),
        IndexOf(search_in="labels", index_of="answerKey", to_field="answer"),
    ],
    task="tasks.qa.multiple_choice.original",
    templates="templates.qa.multiple_choice.no_intro.all",
)
test_card(card)
add_to_catalog(card, "cards.openbook_qa", overwrite=True)
