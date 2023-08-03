from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    RenameFields,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

nli_task = FormTask(
    inputs=["choices", "premise", "hypothesis"],
    outputs=["label"],
    metrics=["metrics.accuracy"],
)

add_to_catalog(nli_task, "tasks.nli", overwrite=True)

nli_templates = TemplatesList(
    [
        InputOutputTemplate(
            input_format="""
                    Given this sentence: {premise}, classify if this sentence: {hypothesis} is {choices}.
                """.strip(),
            output_format="{label}",
        ),
    ]
)

add_to_catalog(nli_templates, "templates.nli", overwrite=True)

card = TaskCard(
    loader=LoadHF(path="glue", name="rte"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "entailment", "1": "not entailment"}}),
        AddFields(
            fields={
                "choices": ["entailment", "not entailment"],
            }
        ),
        RenameFields(
            {
                "sentence1": "premise",
                "sentence2": "hypothesis",
            }
        ),
    ],
    task="tasks.nli",
    templates="templates.nli",
)

test_card(card)
add_to_catalog(card, "cards.rte_concise", overwrite=True)
