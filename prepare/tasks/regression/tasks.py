from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={
            "text": "str",
            "attribute_name": "str",
            "min_value": "Optional[float]",
            "max_value": "Optional[float]",
        },
        outputs={"attribute_value": "float"},
        prediction_type="Any",
        metrics=["metrics.spearman"],
        augmentable_inputs=["text"],
        defaults={"attribute_name": "similarity"},
    ),
    "tasks.regression.single_text",
    overwrite=True,
)

add_to_catalog(
    Task(
        inputs={
            "text1": "str",
            "text2": "str",
            "attribute_name": "str",
            "min_value": "Optional[float]",
            "max_value": "Optional[float]",
        },
        outputs={"attribute_value": "float"},
        prediction_type="Any",
        metrics=["metrics.spearman"],
        augmentable_inputs=["text1", "text2"],
        defaults={"attribute_name": "similarity"},
    ),
    "tasks.regression.two_texts",
    overwrite=True,
)
