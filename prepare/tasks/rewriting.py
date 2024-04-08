from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
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
    ),
    "tasks.rewriting.by_attribute",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["input_text", "text_type"],
        outputs=["output_text"],
        metrics=[
            "metrics.rouge",
        ],
    ),
    "tasks.rewriting.paraphrase",
    overwrite=True,
)
