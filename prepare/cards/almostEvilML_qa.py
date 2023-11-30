from src.unitxt.blocks import FormTask, LoadHF, RenameFields, SplitRandomMix, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import ListFieldValues
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="0x22almostEvil/multilingual-wikihow-qa-16k"),
    preprocess_steps=[
        SplitRandomMix({"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}),
        RenameFields(field_to_field={"INSTRUCTION": "question"}),
        ListFieldValues(fields=["RESPONSE"], to_field="answers"),
    ],
    task=FormTask(
        inputs=["question"],
        outputs=["answers"],
        metrics=["metrics.rouge"],
    ),
    templates="templates.qa.open.all",
)

test_card(card, debug=False)
add_to_catalog(card, "cards.almostEvilML_qa", overwrite=True)
