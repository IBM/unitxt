from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="cnn_dailymail", name="3.0.0"),
    preprocess_steps=[
        RenameFields(field_to_field={"article": "document", "highlights": "summary"}),
        AddFields(fields={"document_type": "article"}),
    ],
    task="tasks.summarization.abstractive",
    templates=TemplatesList(
        [
            "templates.summarization.abstractive.full",
            "templates.summarization.abstractive.write_succinct",
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.cnn_dailymail", overwrite=True)
