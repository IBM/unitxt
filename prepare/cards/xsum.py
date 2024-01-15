from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="EdinburghNLP/xsum"),
    preprocess_steps=[
        AddFields(fields={"document_type": "document"}),
    ],
    task=FormTask(
        inputs=["document", "document_type"],
        outputs=["summary"],
        metrics=["metrics.rouge"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(input_format="{document}", output_format="{summary}"),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.xsum", overwrite=True)
