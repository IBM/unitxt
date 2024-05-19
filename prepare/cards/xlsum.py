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
            "croissant": True,
            "language": "und",
            "language_creators": "unknown",
            "license": "cc-by-nc-sa-4.0",
            "multilinguality": "unknown",
            "region": "us",
            "size_categories": "unknown",
            "source_datasets": "original",
            "task_categories": "summarization",
        },
    )
    if lang == langs[0]:
        test_card(card, debug=False, strict=False)
    add_to_catalog(card, f"cards.xlsum.{lang}", overwrite=True)
