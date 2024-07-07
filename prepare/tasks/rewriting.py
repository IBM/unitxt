from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs=[
            "input_text",
            "input_text_type",
            "required_attribute",
            "output_text_type",
        ],
        outputs=["output_text"],
        metrics=[
            "metrics.rouge",
        ],
        defaults={"input_text_type": "text", "output_text_type": "text"},
    ),
    "tasks.rewriting.by_attribute",
    overwrite=True,
)

add_to_catalog(
    Task(
        inputs=["input_text", "text_type"],
        outputs=["output_text"],
        metrics=[
            "metrics.rouge",
        ],
        defaults={"text_type": "text"},
    ),
    "tasks.rewriting.paraphrase",
    overwrite=True,
)
