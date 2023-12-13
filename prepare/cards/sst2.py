from src.unitxt.blocks import LoadHF, MapInstanceValues, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import AddFields, RenameFields
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="sst2"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "negative", "1": "positive"}}),
        RenameFields(field="sentence", to_field="text"),
        AddFields(
            fields={
                "classes": ["negative", "positive"],
                "text_type": "sentence",
                "type_of_class": "sentiment",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)

test_card(card, debug=True)
add_to_catalog(card, "cards.sst2", overwrite=True)
