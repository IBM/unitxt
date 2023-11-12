from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["text", "text_type", "class"],
        outputs=["class", "label"],
        metrics=[
            "metrics.f1_micro_multi_label",
            "metrics.f1_macro_multi_label",
            "metrics.accuracy",
        ],
    ),
    "tasks.classification.binary",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["text", "text_type", "classes", "type_of_classes"],
        outputs=["labels"],
        metrics=[
            "metrics.f1_micro_multi_label",
            "metrics.accuracy",
            "metrics.f1_macro_multi_label",
        ],
    ),
    "tasks.classification.multi_label",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["text", "text_type", "classes", "type_of_class"],
        outputs=["label"],
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
    ),
    "tasks.classification.multi_class",
    overwrite=True,
)
