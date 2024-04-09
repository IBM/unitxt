from unitxt.blocks import LoadHF, RenameFields, SplitRandomMix, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import ListFieldValues
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="0x22almostEvil/multilingual-wikihow-qa-16k"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}
        ),
        RenameFields(field_to_field={"INSTRUCTION": "question"}),
        ListFieldValues(fields=["RESPONSE"], to_field="answers"),
    ],
    task="tasks.qa.open",
    templates="templates.qa.open.all",
)

test_card(card, debug=False)
add_to_catalog(card, "cards.almost_evil", overwrite=True)
