from src.unitxt.blocks import (
    AddFields,
    LoadFromKaggle,
    MapInstanceValues,
    RenameFields,
    SerializeTableRowAsText,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import ExtractFieldValues
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
                "text_type": "Person",
                "type_of_class": "Heart Disease Possibility",
            }
        ),
        ExtractFieldValues(field="label", to_field="classes", stream_name="train"),
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
            to_field="text",
            max_cell_length=25,
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)

test_card(card, num_demos=3)
add_to_catalog(card, "cards.tablerow_classify", overwrite=True)
