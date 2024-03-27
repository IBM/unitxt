from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"text": "str", "text_type": "str", "class": "str"},
        outputs={"class": "str", "label": "List[str]"},
        prediction_type="str",
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
        inputs={
            "text": "str",
            "text_type": "str",
            "classes": "List[str]",
            "type_of_classes": "str",
        },
        outputs={"labels": "List[str]"},
        prediction_type="str",
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
        inputs={
            "text": "str",
            "text_type": "str",
            "classes": "List[str]",
            "type_of_class": "str",
        },
        outputs={"label": "str"},
        prediction_type="str",
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        augmentable_inputs=["text"],
    ),
    "tasks.classification.multi_class",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={
            "text_a": "str",
            "text_a_type": "str",
            "text_b": "str",
            "text_b_type": "str",
            "classes": "List[str]",
            "type_of_relation": "str",
        },
        outputs={"label": "str"},
        prediction_type="str",
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        augmentable_inputs=["text_a", "text_b"],
    ),
    "tasks.classification.multi_class.relation",
    overwrite=True,
)
