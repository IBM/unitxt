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
    )
    if lang == langs[0]:
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.mlsum.{lang}", overwrite=True)
