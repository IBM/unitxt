from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
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
    ),
    "tasks.regression.single_text",
    overwrite=True,
)

add_to_catalog(
    FormTask(
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
    ),
    "tasks.regression.two_texts",
    overwrite=True,
)
