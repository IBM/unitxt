from src.unitxt import add_to_catalog
from src.unitxt.blocks import (
    AddFields,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.loaders import LoadCSV
from src.unitxt.test_utils.card import test_card

dataset_name = "CFPB_Product"


card = TaskCard(
    loader=LoadCSV(
        files={
            "train": "https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/cfpb_complaints/cfpb_compliants.csv"
        }
    ),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[70%]", "validation": "train[10%]", "test": "train[20%]"}
        ),
        RenameFields(field_to_field={"narrative": "text", "product": "label"}),
        AddFields(
            fields={
                "classes": [
                    "retail_banking",
                    "mortgages_and_loans",
                    "debt_collection",
                    "credit_card",
                    "credit_reporting",
                ],
                "text_type": "text",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}.5_classes", overwrite=True)
