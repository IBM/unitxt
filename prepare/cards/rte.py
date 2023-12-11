from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

nli_task = FormTask(
    inputs=["choices", "premise", "hypothesis"],
    outputs=["label"],
    metrics=["metrics.accuracy"],
)

add_to_catalog(nli_task, "tasks.nli", overwrite=True)

card = TaskCard(
    loader=LoadHF(path="glue", name="rte"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "choices": ["entailment", "not entailment"],
            }
        ),
        RenameFields(
            field_to_field={
                "sentence1": "premise",
                "sentence2": "hypothesis",
            }
        ),
    ],
    task="tasks.nli",
    templates="templates.classification.nli.all",
)

test_card(card)
add_to_catalog(card, "cards.rte", overwrite=True)
