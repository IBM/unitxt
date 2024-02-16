from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadHFCustomDatasetScript,
    MapInstanceValues,
    SerializeTableAsIndexedRowMajor,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList
from src.unitxt.test_utils.card import test_card

# Set unitxt.settings.allow_unverified_code=True or environment vairable: UNITXT_ALLOW_UNVERIFIED_CODE to True

card = TaskCard(
    loader=LoadHFCustomDatasetScript(file="tabfact.py", streaming=False),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "refuted", "1": "entailed"}}),
        AddFields(
            fields={
                "choices": ["refuted", "entailed"],
            }
        ),
        SerializeTableAsIndexedRowMajor(field_to_field=[["table", "table_serialized"]]),
    ],
    task=FormTask(
        inputs=["table_serialized", "statement", "choices"],
        outputs=["label"],
        metrics=[
            "metrics.accuracy",
        ],
        augmentable_inputs=["statement"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="Given this table: {table_serialized}, classify if this sentence '{statement}' is {choices}? ",
                output_format="{label}",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                ],
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.tabfact", overwrite=True)
