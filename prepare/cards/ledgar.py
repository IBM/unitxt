from datasets import load_dataset_builder
from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    TaskCard,
)
from unitxt.test_utils.card import test_card

dataset_name = "ledgar"

ds_builder = load_dataset_builder("lex_glue", dataset_name)
classlabels = ds_builder.info.features["label"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]

card = TaskCard(
    loader=LoadHF(path="lex_glue", name=f"{dataset_name}"),
    preprocess_steps=[
        MapInstanceValues({"label": mappers}),
        AddFields(
            fields={
                "classes": classlabels.names,
                "text_type": "text",
                "type_of_class": "contractual clauses",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:question-answering",
            "task_categories:text-classification",
            "task_ids:multi-class-classification",
            "task_ids:multi-label-classification",
            "task_ids:multiple-choice-qa",
            "task_ids:topic-classification",
            "annotations_creators:found",
            "language_creators:found",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:extended",
            "language:en",
            "license:cc-by-4.0",
            "croissant",
            "arxiv:2110.00976",
            "arxiv:2109.00904",
            "arxiv:1805.01217",
            "arxiv:2104.08671",
            "region:us",
        ]
    },
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
