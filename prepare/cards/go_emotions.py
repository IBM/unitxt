from datasets import load_dataset_builder
from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    Set,
    TaskCard,
)
from unitxt.operators import MapInstanceValues
from unitxt.test_utils.card import test_card

dataset_name = "go_emotions"
subset = "simplified"

ds_builder = load_dataset_builder(dataset_name, subset)
classes = ds_builder.info.features["labels"].feature.names

mappers = {str(i): cls for i, cls in enumerate(classes)}

card = TaskCard(
    loader=LoadHF(path=dataset_name, name=subset),
    preprocess_steps=[
        MapInstanceValues(mappers={"labels": mappers}, process_every_value=True),
        Set(
            fields={
                "classes": classes,
                "text_type": "text",
                "type_of_classes": "emotions",
            }
        ),
    ],
    task="tasks.classification.multi_label",
    templates="templates.classification.multi_label.all",
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": "2005.00547",
        "language": "en",
        "language_creators": "found",
        "license": "apache-2.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": ["multi-class-classification", "multi-label-classification"],
    },
    __description__=(
        "The GoEmotions dataset contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral. The raw data is included as well as the smaller, simplified version of the dataset with predefined train/val/test splits… See the full description on the dataset page: https://huggingface.co/datasets/go_emotions."
    ),
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}.{subset}", overwrite=True)
