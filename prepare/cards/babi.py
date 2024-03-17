from src.unitxt.blocks import AddFields, LoadHF, RenameFields, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import ListFieldValues
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="Muennighoff/babi"),
    preprocess_steps=[
        RenameFields(field_to_field={"passage": "context"}),
        ListFieldValues(fields=["answer"], to_field="answers"),
        AddFields({"context_type": "description"}),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
)

test_card(card)
add_to_catalog(card, "cards.babi.qa", overwrite=True)
