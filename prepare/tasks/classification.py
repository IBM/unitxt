from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["text", "text_type", "class"],
        outputs=["class", "label"],
        metrics=[
            "metrics.f1_micro_multi_label",
            "metrics.f1_macro_multi_label",
            "metrics.accuracy",
        ],
        augmentable_inputs=["text"],
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
        augmentable_inputs=["text"],
    ),
    "tasks.classification.multi_label",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["text", "text_type", "classes", "type_of_class"],
        outputs=["label"],
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        augmentable_inputs=["text"],
    ),
    "tasks.classification.multi_class",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=[
            "text_a",
            "text_a_type",
            "text_b",
            "text_b_type",
            "classes",
            "type_of_relation",
        ],
        outputs=["label"],
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        augmentable_inputs=["text_a", "text_b"],
    ),
    "tasks.classification.multi_class.relation",
    overwrite=True,
)
