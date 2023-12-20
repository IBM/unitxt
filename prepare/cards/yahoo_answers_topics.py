from datasets import load_dataset_builder

from src.unitxt import add_to_catalog
from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.operators import JoinStr, ListFieldValues
from src.unitxt.test_utils.card import test_card

dataset_name = "yahoo_answers_topics"

ds_builder = load_dataset_builder(dataset_name)
classlabels = ds_builder.info.features["topic"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]


card = TaskCard(
    loader=LoadHF(path=f"{dataset_name}"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[87.5%]", "validation": "train[12.5%]", "test": "test"}
        ),
        RenameFields(field_to_field={"topic": "label"}),
        MapInstanceValues(mappers={"label": mappers}),
        ListFieldValues(
            fields=["question_title", "question_content", "best_answer"],
            to_field="text",
        ),
        JoinStr(separator=" ", field="text", to_field="text"),
        AddFields(
            fields={
                "classes": classlabels.names,
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
