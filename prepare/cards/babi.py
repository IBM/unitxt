from unitxt.blocks import AddFields, LoadHF, RenameFields, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import ListFieldValues
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="Muennighoff/babi"),
    preprocess_steps=[
        RenameFields(field_to_field={"passage": "context"}),
        AddFields({"context_type": "description"}),
        ListFieldValues(fields=["answer"], to_field="answers"),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
    __tags__={"croissant": True, "region": "us"},
)

test_card(card)
add_to_catalog(card, "cards.babi.qa", overwrite=True)
