from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=[
            "required_attribute",
            "attribute_type",
            "choices_texts",
            "choices_text_type",
        ],
        outputs=["choices_texts", "choice"],
        metrics=[
            "metrics.accuracy",
        ],
    ),
    "tasks.selection.by_attribute",
    overwrite=True,
)
