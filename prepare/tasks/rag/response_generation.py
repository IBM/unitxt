from typing import List, Union

from unitxt import add_to_catalog
from unitxt.blocks import (
    Task,
)

add_to_catalog(
    Task(
        input_fields={
            "contexts": List[str],
            "contexts_ids": Union[List[int], List[str]],
            "question": str,
        },
        reference_fields={"reference_answers": List[str]},
        metrics=[
            "metrics.rag.response_generation.correctness.token_overlap",
            "metrics.rag.response_generation.faithfullness.token_overlap",
            "metrics.rag.response_generation.correctness.bert_score.deberta_large_mnli",
        ],
        augmentable_inputs=["contexts", "question"],
    ),
    "tasks.rag.response_generation",
    overwrite=True,
)
