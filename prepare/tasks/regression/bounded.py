from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["text", "value", "type_of_value", "min_value", "max_value"],
        outputs=["value"],
        metrics=["metrics.spearman"],
        augmentable_inputs=["text"],
    ),
    "tasks.regression.bounded.single",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["text1", "text2", "type_of_value", "min_value", "max_value"],
        outputs=["value"],
        metrics=["metrics.spearman"],
        augmentable_inputs=["text1", "text2"],
    ),
    "tasks.regression.bounded.pair",
    overwrite=True,
)
