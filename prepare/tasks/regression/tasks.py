from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["text", "attribute_name", "min_value", "max_value"],
        outputs=["attribute_value"],
        metrics=["metrics.spearman"],
        augmentable_inputs=["text"],
    ),
    "tasks.regression.single_text",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["text1", "text2", "attribute_name", "min_value", "max_value"],
        outputs=["attribute_value"],
        metrics=["metrics.spearman"],
        augmentable_inputs=["text1", "text2"],
    ),
    "tasks.regression.two_texts",
    overwrite=True,
)
