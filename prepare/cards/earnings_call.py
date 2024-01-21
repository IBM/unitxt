from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    InputOutputTemplate,
    LoadHF,
    TaskCard,
    TemplatesList,
)
from unitxt.test_utils.card import test_card

"""
TaskCard generated from HELM Enterprise Scenario:

- earningscall_scenario.py

https://github.ibm.com/ai-models-evaluation/crfm-helm-enterprise

"""
card = TaskCard(
    loader=LoadHF(path="jlh-ibm/earnings_call"),
    preprocess_steps=[
        AddFields(
            fields={
                "text_type": "earning call",
                "classes": ["positive", "negative"],
                "type_of_class": "sentiment",
            }
        )
    ],
    task="tasks.classification.multi_class",
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="{text}\nQuestion: Classify the above paragraph into one of the following sentiments: "
                "negative/positive.",
                output_format="{label}",
            )
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.earnings_call", overwrite=True)
