from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={"document": "str", "document_type": "str"},
        reference_fields={"summary": "str"},
        prediction_type="str",
        metrics=["metrics.rouge"],
        defaults={"document_type": "document"},
    ),
    "tasks.summarization.abstractive",
    overwrite=True,
)
