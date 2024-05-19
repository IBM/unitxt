from datasets import get_dataset_config_names
from unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

langs = get_dataset_config_names("mlsum")  # the languages


for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="mlsum", name=lang),
        preprocess_steps=[
            RenameFields(field_to_field={"text": "document"}),
            AddFields(fields={"document_type": "document"}),
        ],
        task="tasks.summarization.abstractive",
        templates="templates.summarization.abstractive.all",
        __tags__={
            "annotations_creators": "found",
            "croissant": True,
            "language": ["de", "es", "fr", "ru", "tr"],
            "language_creators": "found",
            "license": "other",
            "multilinguality": "multilingual",
            "region": "us",
            "size_categories": ["100K<n<1M", "10K<n<100K"],
            "source_datasets": ["extended|cnn_dailymail", "original"],
            "task_categories": ["summarization", "translation", "text-classification"],
            "task_ids": [
                "news-articles-summarization",
                "multi-class-classification",
                "multi-label-classification",
                "topic-classification",
            ],
        },
    )
    if lang == langs[0]:
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.mlsum.{lang}", overwrite=True)
