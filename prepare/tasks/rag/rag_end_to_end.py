from typing import Any, Dict, List

from unitxt import add_to_catalog
from unitxt.blocks import Task

add_to_catalog(
    Task(
        input_fields={
            "question": str,
            "question_id": Any,
            "metadata_field": str,
        },
        reference_fields={
            "reference_answers": List[str],
            "reference_contexts": List[str],
            "reference_context_ids": List[str],
            "is_answerable_label": bool,
        },
        metrics=[
            "metrics.rag.end_to_end.answer_correctness",
            "metrics.rag.end_to_end.answer_faithfulness",
            "metrics.rag.end_to_end.answer_reward",
            "metrics.rag.end_to_end.context_correctness",
            "metrics.rag.end_to_end.context_relevance",
        ],
        prediction_type=Dict[str, Any],
        augmentable_inputs=["question"],
    ),
    "tasks.rag.end_to_end",
    overwrite=True,
)

add_to_catalog(
    Task(
        input_fields={
            "document_id": str,
            "title": str,
            "passages": List[str],
            "metadata_field": str,
        },
        reference_fields={},
        prediction_type=Any,
        metrics=[
            "metrics.rouge"
        ],  # We can not define an empty metric, so we gave here a simple one- although rouge is not related
    ),
    "tasks.rag.corpora",
    overwrite=True,
)
