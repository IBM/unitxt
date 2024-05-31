import sys

from datasets import get_dataset_config_names, load_dataset_builder
from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from unitxt.operators import Shuffle
from unitxt.test_utils.card import test_card

dataset_name = "clinc_oos"

for subset in get_dataset_config_names(dataset_name):
    ds_builder = load_dataset_builder(dataset_name, subset)
    classlabels = ds_builder.info.features["intent"]
    map_label_to_text = {
        str(i): label.replace("_", " ") for i, label in enumerate(classlabels.names)
    }
    classes = [label.replace("_", " ") for label in classlabels.names]

    card = TaskCard(
        loader=LoadHF(path=dataset_name, name=subset),
        preprocess_steps=[
            Shuffle(page_size=sys.maxsize),
            RenameFields(field_to_field={"intent": "label"}),
            MapInstanceValues(mappers={"label": map_label_to_text}),
            AddFields(
                fields={
                    "classes": classes,
                    "text_type": "sentence",
                    "type_of_class": "intent",
                }
            ),
        ],
        task="tasks.classification.multi_class",
        templates="templates.classification.multi_class.all",
        __tags__={
            "annotations_creators": "expert-generated",
            "language": "en",
            "language_creators": "crowdsourced",
            "license": "cc-by-3.0",
            "multilinguality": "monolingual",
            "region": "us",
            "singletons": ["croissant"],
            "size_categories": "10K<n<100K",
            "source_datasets": "original",
            "task_categories": "text-classification",
            "task_ids": "intent-classification",
        },
        __description__=(
            "Dataset Card for CLINC150\n"
            "Dataset Summary\n"
            "Task-oriented dialog systems need to know when a query falls outside their range of supported intents, but current text classification corpora only define label sets that cover every example. We introduce a new dataset that includes queries that are out-of-scope (OOS), i.e., queries that do not fall into any of the system's supported intents. This poses a new challenge because models cannot assume that every query at… See the full description on the dataset page: https://huggingface.co/datasets/clinc_oos."
        ),
    )
    test_card(card, debug=False)
    add_to_catalog(artifact=card, name=f"cards.{dataset_name}.{subset}", overwrite=True)
