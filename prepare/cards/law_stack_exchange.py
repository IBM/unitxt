from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import JoinStr, ListFieldValues
from unitxt.test_utils.card import test_card

dataset_name = "law_stack_exchange"


huggingface_name = "jonathanli/law-stack-exchange"

classlabels = [
    "business",
    "civil-law",
    "constitutional-law",
    "contract",
    "contract-law",
    "copyright",
    "criminal-law",
    "employment",
    "intellectual-property",
    "internet",
    "liability",
    "licensing",
    "privacy",
    "software",
    "tax-law",
    "trademark",
]


card = TaskCard(
    loader=LoadHF(path=huggingface_name),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "test", "test": "train", "validation": "validation"}
        ),  # TODO consider switch between test and train
        RenameFields(field_to_field={"text_label": "label"}),
        ListFieldValues(fields=["title", "body"], to_field="text"),
        JoinStr(separator=". ", field="text", to_field="text"),
        AddFields(
            fields={
                "classes": classlabels,
                "text_type": "text",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
