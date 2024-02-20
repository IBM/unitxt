from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadFromKaggle,
    MapInstanceValues,
    RenameFields,
    SerializeTableRowAsText,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadFromKaggle(
        url="https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"
    ),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[70%]", "validation": "train[10%]", "test": "train[20%]"}
        ),
        RenameFields(field_to_field={"HeartDisease": "label"}),
        MapInstanceValues(mappers={"label": {"0": "Normal", "1": "Heart Disease"}}),
        AddFields(
            fields={
                "choices": ["Normal", "Heart Disease"],
            }
        ),
        SerializeTableRowAsText(
            fields=[
                "Age",
                "Sex",
                "ChestPainType",
                "RestingBP",
                "Cholesterol",
                "FastingBS",
                "RestingECG",
                "MaxHR",
                "ExerciseAngina",
                "Oldpeak",
                "ST_Slope",
            ],
            to_field="serialized_row",
            max_cell_length=25,
        ),
    ],
    task=FormTask(
        inputs=["serialized_row", "choices"],
        outputs=["label"],
        metrics=["metrics.accuracy"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given the following details of a person {serialized_row} we need to predict the possibility of heart disease for the person. Classify if the person is {choices}
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),
)

test_card(card, num_demos=3)
add_to_catalog(card, "cards.tablerow_classify", overwrite=True)
