from datasets import get_dataset_config_names
from unitxt.blocks import (
    LoadHF,
    Rename,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.settings_utils import get_settings
from unitxt.test_utils.card import test_card

settings = get_settings()

langs = get_dataset_config_names(
    "mlsum", trust_remote_code=settings.allow_unverified_code
)  # the languages


for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="mlsum", name=lang),
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
            "MLSUM is a large-scale MultiLingual SUMmarization dataset. Obtained from online newspapers, it contains 1.5M+ article/summary pairs in five different languages: French, German, Spanish, Russian, Turkish. Together with English newspapers from the popular CNN/Daily mail dataset, the collected data form a large scale multilingual dataset which can enable new research directions for the text summarization community."
        ),
        __short_description__=(
            "Uses a dataset with over 1.5 M article-and-summary pairs from online newspapers in 5 languages (French, German, Spanish, Russian, Turkish) and English newspapers from CNN and Daily Mail to evaluate a model's ability to summarize multilingual text. Metric shows the ROUGE-L score for the generated summary."
        ),
    )
    if lang == langs[0]:
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.mlsum.{lang}", overwrite=True)
