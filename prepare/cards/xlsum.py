from datasets import get_dataset_config_names
from unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

configs = get_dataset_config_names("GEM/xlsum")  # the languages
# now configs is the list of all languages showing in the dataset


langs = configs

for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="GEM/xlsum", name=lang),
        preprocess_steps=[
            RenameFields(field_to_field={"text": "document", "target": "summary"}),
            AddFields(fields={"document_type": "document"}),
        ],
        task="tasks.summarization.abstractive",
        templates="templates.summarization.abstractive.all",
        __tags__={
            "annotations_creators": "none",
            "arxiv": "1607.01759",
            "language": "und",
            "language_creators": "unknown",
            "license": "cc-by-nc-sa-4.0",
            "multilinguality": "unknown",
            "region": "us",
            "singletons": ["croissant"],
            "size_categories": "unknown",
            "source_datasets": "original",
            "task_categories": "summarization",
        },
        __description__=(
            "We present XLSum, a comprehensive and diverse dataset comprising 1.35 million professionally\n"
            "annotated article-summary pairs from BBC, extracted using a set of carefully designed heuristics.\n"
            "The dataset covers 45 languages ranging from low to high-resource, for many of which no\n"
            "public dataset is currently available. XL-Sum is highly abstractive, concise,\n"
            "and of high quality, as indicated by human and intrinsic evaluation."
        ),
    )
    if lang == langs[0]:
        test_card(card, debug=False, strict=False)
    add_to_catalog(card, f"cards.xlsum.{lang}", overwrite=True)
