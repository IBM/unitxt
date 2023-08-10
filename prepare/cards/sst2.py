import datasets as ds
from src.unitxt import dataset
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import RenameFields
from src.unitxt.splitters import RenameSplits
from src.unitxt.test_utils.card import test_card

one_sentence_classification_templates = TemplatesList(
    [
        InputOutputTemplate(
            input_format="""
                    Given this sentence: {sentence}, classify if it is {choices}.
                """.strip(),
            output_format="{label}",
        ),
    ]
)
add_to_catalog(one_sentence_classification_templates, "templates.one_sent_classification", overwrite=True)

one_sentence_classification_task = FormTask(
    inputs=["choices", "sentence"],
    outputs=["label"],
    metrics=["metrics.accuracy"],
)
add_to_catalog(one_sentence_classification_task, "tasks.one_sent_classification", overwrite=True)


card = TaskCard(
    loader=LoadHF(path="glue", name="sst2"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "negative", "1": "positive"}}),
        AddFields(
            fields={
                "choices": ["negative", "positive"],
            }
        ),
    ],
    task="tasks.one_sent_classification",
    templates="templates.one_sent_classification",
)

test_card(card)
add_to_catalog(card, "cards.sst2", overwrite=True)
