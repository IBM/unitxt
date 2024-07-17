from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields=[
            "required_attribute",
            "attribute_type",
            "choices_texts",
            "choices_text_type",
        ],
        reference_fields=["choices_texts", "choice"],
        metrics=[
            "metrics.accuracy",
        ],
    ),
    "tasks.selection.by_attribute",
    overwrite=True,
)
