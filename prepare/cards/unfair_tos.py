from datasets import load_dataset_builder

from src.unitxt import add_to_catalog
from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    MultiLabelTemplate,
    TaskCard,
    TemplatesList,
)
from src.unitxt.test_utils.card import test_card

dataset_name = "unfair_tos"

ds_builder = load_dataset_builder("lex_glue", dataset_name)
classlabels = ds_builder.info.features["labels"]

mappers = {}
for i in range(len(classlabels.feature.names)):
    mappers[str(i)] = classlabels.feature.names[i]

card = TaskCard(
    loader=LoadHF(path="lex_glue", name=f"{dataset_name}"),
    preprocess_steps=[
        MapInstanceValues(mappers={"labels": mappers}, process_every_value=True),
        AddFields(
            fields={
                "classes": classlabels.feature.names,
                "text_type": "text",
                "type_of_classes": "contractual clauses",
            }
        ),
    ],
    task="tasks.classification.multi_label",
    templates=TemplatesList(
        [
            MultiLabelTemplate(
                input_format="{text}",
                output_format="{labels}",
            ),
        ]
    ),
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
