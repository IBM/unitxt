import sys

from datasets import load_dataset_builder
from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import Shuffle
from unitxt.test_utils.card import test_card

dataset_name = "banking77"

ds_builder = load_dataset_builder(dataset_name)
classlabels = ds_builder.info.features["label"]

map_label_to_text = {
    str(i): label.replace("_", " ") for i, label in enumerate(classlabels.names)
}
classes = [label.replace("_", " ") for label in classlabels.names]

card = TaskCard(
    loader=LoadHF(path=f"PolyAI/{dataset_name}"),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        SplitRandomMix(
            {"train": "train[85%]", "validation": "train[15%]", "test": "test"}
        ),
        MapInstanceValues(mappers={"label": map_label_to_text}),
        AddFields(
            fields={
                "classes": classes,
                "text_type": "utterance",
                "type_of_class": "intent",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)
test_card(card, debug=False)
add_to_catalog(artifact=card, name=f"cards.{dataset_name}", overwrite=True)
