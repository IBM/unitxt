from datasets import load_dataset_builder

from src.unitxt import add_to_catalog
from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from src.unitxt.test_utils.card import test_card

dataset_name = "clinc150"
subset = "plus"  # TODO add imbalanced, small
ds_builder = load_dataset_builder("clinc_oos", subset)
classlabels = ds_builder.info.features["intent"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]


card = TaskCard(
    loader=LoadHF(path="clinc_oos", name=subset),
    preprocess_steps=[
        RenameFields(field_to_field={"intent": "label"}),
        MapInstanceValues(mappers={"label": mappers}),
        AddFields(
            fields={
                "classes": mappers,
                "text_type": "sentence",
                "type_of_class": "intent",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
)
test_card(card, debug=False)
add_to_catalog(artifact=card, name=f"cards.{dataset_name}.{subset}", overwrite=True)
