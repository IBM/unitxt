from unitxt.blocks import (
    AddFields,
    LoadHF,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="EdinburghNLP/xsum"),
    preprocess_steps=[
        AddFields(fields={"document_type": "document"}),
    ],
    task="tasks.summarization.abstractive",
    templates="templates.summarization.abstractive.all",
)

test_card(card)
add_to_catalog(card, "cards.xsum", overwrite=True)
