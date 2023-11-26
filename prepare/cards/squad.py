from src.unitxt.blocks import CopyFields, FormTask, LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="squad"),
    preprocess_steps=[
        "splitters.small_no_test",
        CopyFields(field_to_field=[["answers/text", "answer"]], use_query=True),
    ],
    task=FormTask(
        inputs=["context", "question"],
        outputs=["answer"],
        metrics=["metrics.squad"],
    ),
    templates="templates.qa.contextual.all",
)

test_card(card)
add_to_catalog(card, "cards.squad", overwrite=True)
