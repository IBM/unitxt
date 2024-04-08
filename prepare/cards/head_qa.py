from datasets import get_dataset_config_names
from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    TaskCard,
)
from unitxt.test_utils.card import test_card

dataset_name = "head_qa"

categories = [
    "biology",
    "chemistry",
    "medicine",
    "nursery",
    "pharmacology",
    "psychology",
]
for subset in get_dataset_config_names(dataset_name):
    card = TaskCard(
        loader=LoadHF(path=f"{dataset_name}", name=subset),
        preprocess_steps=[
            RenameFields(field_to_field={"qtext": "text", "category": "label"}),
            AddFields(
                fields={
                    "classes": categories,
                    "text_type": "question",
                    "type_of_class": "topic",
                }
            ),
        ],
        task="tasks.classification.multi_class",
        templates="templates.classification.multi_class.all",
    )
    test_card(card, debug=False)
    add_to_catalog(card, f"cards.{dataset_name}.{subset}", overwrite=True)
