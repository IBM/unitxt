from datasets import load_dataset_builder
from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    SplitRandomMix,
    TaskCard,
)
from unitxt.test_utils.card import test_card

dataset_name = "ag_news"

ds_builder = load_dataset_builder(dataset_name)
classlabels = ds_builder.info.features["label"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]


card = TaskCard(
    loader=LoadHF(path=f"{dataset_name}"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[87.5%]", "validation": "train[12.5%]", "test": "test"}
        ),
        MapInstanceValues(mappers={"label": mappers}),
        AddFields(
            fields={
                "classes": classlabels.names,
                "text_type": "sentence",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
