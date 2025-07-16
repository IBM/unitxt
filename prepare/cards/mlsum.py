from unitxt.blocks import (
    LoadHF,
    Rename,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.test_utils.card import test_card

langs = ["de", "es", "fr", "ru", "tu"]

for lang in langs:
    card = TaskCard(
        loader=LoadHF(
            path="mlsum",
            revision="refs/convert/parquet",
            data_dir=lang,
            splits=["train", "test", "validation"],
        ),
        preprocess_steps=[
            Rename(field_to_field={"text": "document"}),
            Wrap(field="summary", inside="list", to_field="summaries"),
        ],
        task="tasks.summarization.abstractive",
        templates="templates.summarization.abstractive.all",
        __tags__={
            "annotations_creators": "found",
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
        __description__=(
            "We present MLSUM, the first large-scale MultiLingual SUMmarization dataset.\n"
            "Obtained from online newspapers, it contains 1.5M+ article/summary pairs in five different languages -- namely, French, German, Spanish, Russian, Turkish.\n"
            "Together with English newspapers from the popular CNN/Daily mail dataset, the collected data form a large scale multilingual dataset which can enable new research directions for the text summarization community.\n"
            "We report cross-lingual comparative analyses based on state-of-the-art systems.\n"
            "These highlight existing biases which motivate the use of a multi-lingual dataset."
        ),
    )
    if lang == langs[-1]:
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.mlsum.{lang}", overwrite=True)
