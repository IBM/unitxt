from src.unitxt.blocks import (
    AddFields,
    CopyFields,
    FormTask,
    LoadHF,
    SpanLabelingTemplate,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="/dccstor/actr/unitxt-datasets/entities_selected_new"),
    preprocess_steps=[
        CopyFields(
            field_to_field={
                "mentions/*/location/begin": "spans_starts",
                "mentions/*/location/end": "spans_ends",
                "mentions/*/type": "text",
            }
        ),
        AddFields(
            {
                "choices": [
                    "Person",
                    "Measure",
                    "Number",
                    "Facility",
                    "Location",
                    "Product",
                    "Duration",
                    "Money",
                    "Time",
                    "PhoneNumber",
                    "Date",
                    "JobTitle",
                    "Organization",
                    "Percent",
                    "GeographicFeature",
                    "Address",
                    "Ordinal",
                ]
            }
        ),
    ],
    task=FormTask(
        inputs=["text", "choices"],
        outputs=["spans_starts", "spans_ends", "text", "labels"],
        metrics=["metrics.spearman"],
    ),
    templates=TemplatesList(
        [
            SpanLabelingTemplate(
                input_format="""
                   Given this named entities: {choices}, mark them in the text: {text}
                """.strip(),
                output_format="{spans_and_labels}",
                span_label_format="{span}: {label}",
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.stsb", overwrite=True)
