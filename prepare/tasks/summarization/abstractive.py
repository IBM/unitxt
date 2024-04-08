from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["document", "document_type"],
        outputs=["summary"],
        metrics=["metrics.rouge"],
    ),
    "tasks.summarization.abstractive",
    overwrite=True,
)
