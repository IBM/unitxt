from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"document": "str", "document_type": "str"},
        outputs={"summary": "str"},
        prediction_type="str",
        metrics=["metrics.rouge"],
    ),
    "tasks.summarization.abstractive",
    overwrite=True,
)
