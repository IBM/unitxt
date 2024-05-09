from dataclasses import dataclass

from unitxt import add_to_catalog
from unitxt.blocks import (
    FormTask,
)

add_to_catalog(
    FormTask(
        inputs=["contexts", "contexts_ids", "question"],
        outputs=["reference_answers"],
        metrics=[
            "metrics.rag.response_generation.correctness.token_overlap",
            "metrics.rag.response_generation.faithfullness.token_overlap",
            "metrics.rag.response_generation.correctness.bert_score.deberta_large_mnli",
        ],
        augmentable_inputs=["contexts", "question"],
    ),
    "tasks.rag.response_generation",
)

