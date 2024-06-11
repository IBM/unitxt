from unitxt import add_to_catalog
from unitxt.blocks import (
    MapInstanceValues,
    RenameFields,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.loaders import LoadCSV
from unitxt.test_utils.card import test_card

dataset_name = "medical_abstracts"


mappers = {
    "1": "neoplasms",
    "2": "digestive system diseases",
    "3": "nervous system diseases",
    "4": "cardiovascular diseases",
    "5": "general pathological conditions",
}

card = TaskCard(
    loader=LoadCSV(
        files={
            "train": "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv",
            "test": "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv",
        }
    ),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[90%]", "validation": "train[10%]", "test": "test"}
        ),
        RenameFields(
            field_to_field={"medical_abstract": "text", "condition_label": "label"}
        ),
        MapInstanceValues(mappers={"label": mappers}),
        Set(
            fields={
                "classes": list(mappers.values()),
                "text_type": "abstract",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
