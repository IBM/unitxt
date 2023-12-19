from datasets import load_dataset_builder

from src.unitxt import add_to_catalog
from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.test_utils.card import test_card

dataset_name = "banking77"

ds_builder = load_dataset_builder(dataset_name)
classlabels = ds_builder.info.features["label"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]


card = TaskCard(
    loader=LoadHF(path=f"PolyAI/{dataset_name}"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[85%]", "validation": "train[15%]", "test": "test"}
        ),
        MapInstanceValues(mappers={"label": mappers}),
        AddFields(
            fields={
                "classes": classlabels.names,
                "text_type": "sentence",
                "type_of_class": "intent",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)
test_card(card, debug=False)
add_to_catalog(artifact=card, name=f"cards.{dataset_name}", overwrite=True)
