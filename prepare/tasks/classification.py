from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={"text": "str", "text_type": "str", "class": "str"},
        outputs={"class": "str", "label": "List[str]"},
        prediction_type="List[str]",
        metrics=[
            "metrics.f1_micro_multi_label",
            "metrics.f1_macro_multi_label",
            "metrics.accuracy",
        ],
        augmentable_inputs=["text"],
        defaults={"text_type": "text"},
    ),
    "tasks.classification.binary",
    overwrite=True,
)

add_to_catalog(
    Task(
        inputs={"text": "str", "text_type": "str", "class": "str"},
        outputs={"class": "str", "label": "int"},
        prediction_type="float",
        metrics=[
            "metrics.accuracy",
            "metrics.f1_binary",
        ],
        augmentable_inputs=["text"],
        defaults={"text_type": "text"},
    ),
    "tasks.classification.binary.zero_or_one",
    overwrite=True,
)

add_to_catalog(
    Task(
        inputs={
            "text": "str",
            "text_type": "str",
            "classes": "List[str]",
            "type_of_classes": "str",
        },
        outputs={"labels": "List[str]"},
        prediction_type="List[str]",
        metrics=[
            "metrics.f1_micro_multi_label",
            "metrics.accuracy",
            "metrics.f1_macro_multi_label",
        ],
        augmentable_inputs=["text"],
        defaults={"text_type": "text"},
    ),
    "tasks.classification.multi_label",
    overwrite=True,
)

add_to_catalog(
    Task(
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
        defaults={"text_type": "text"},
    ),
    "tasks.classification.multi_class",
    overwrite=True,
)

add_to_catalog(
    Task(
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
        defaults={"text_a_type": "first text", "text_b_type": "second text"},
    ),
    "tasks.classification.multi_class.relation",
    overwrite=True,
)


add_to_catalog(
    Task(
        inputs={
            "text": "str",
            "text_type": "str",
            "classes": "List[str]",
            "type_of_class": "str",
            "classes_descriptions": "str",
        },
        outputs={"label": "str"},
        prediction_type="str",
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        augmentable_inputs=["text"],
        defaults={"text_type": "text", "type_of_class": "topic"},
    ),
    "tasks.classification.multi_class.with_classes_descriptions",
    overwrite=True,
)

add_to_catalog(
    Task(
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
        defaults={"text_type": "text", "type_of_class": "topic"},
    ),
    "tasks.classification.multi_class.topic_classification",
    overwrite=True,
)
