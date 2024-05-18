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
            "dataset_info_tags": [
                "task_categories:summarization",
                "task_categories:translation",
                "task_categories:text-classification",
                "task_ids:news-articles-summarization",
                "task_ids:multi-class-classification",
                "task_ids:multi-label-classification",
                "task_ids:topic-classification",
                "annotations_creators:found",
                "language_creators:found",
                "multilinguality:multilingual",
                "size_categories:100K<n<1M",
                "size_categories:10K<n<100K",
                "source_datasets:extended|cnn_dailymail",
                "source_datasets:original",
                "language:de",
                "language:es",
                "language:fr",
                "language:ru",
                "language:tr",
                "license:other",
                "croissant",
                "region:us",
            ]
        },
    )
    if lang == langs[0]:
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.mlsum.{lang}", overwrite=True)
