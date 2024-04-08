from unitxt.logging_utils import get_logger

"""TaskCard generated from HELM Enterprise Scenario:

- earningscall_scenario.py

https://github.ibm.com/ai-models-evaluation/crfm-helm-enterprise

"""
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
"""

get_logger().info(
    "earning_call.py card is disabled due to a bug in the Hugginface dataset."
    "Waiting for a fix to be issue. "
    "PR at https://huggingface.co/datasets/jlh-ibm/earnings_call/discussions/2"
)
