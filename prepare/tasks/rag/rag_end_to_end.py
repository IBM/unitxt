from typing import Any, Dict, List, Union

from unitxt import add_to_catalog
from unitxt.blocks import Task
from unitxt.types import Dialog, Image, RagResponse

add_to_catalog(
    Task(
        __description__="""This is a task corresponding to an end to end RAG evaluation.  It assumes the user provides a question, and
        the RAG system returns an answer and a set of retrieved contexts (documents or passages).
        For details of RAG see: https://www.unitxt.ai/en/latest/docs/rag_support.html.
""",
        input_fields={
            "question": Union[str, Dialog],
            "question_id": Any,
            "metadata_tags": Dict[str, str],
        },
        reference_fields={
            "reference_answers": List[str],
            "reference_contexts": List[str],
            "reference_context_ids": Union[List[int], List[str]],
            "reference_context_page_ids": List[int],
            "is_answerable_label": bool,
        },
        metrics=[
            "metrics.rag.end_to_end.answer_correctness",
            "metrics.rag.end_to_end.answer_faithfulness",
            "metrics.rag.end_to_end.answer_reward",
            "metrics.rag.end_to_end.context_correctness",
            "metrics.rag.end_to_end.context_relevance",
        ],
        prediction_type=RagResponse,
        augmentable_inputs=["question"],
        defaults={
            "question_id": "",
            "metadata_tags": {},
            "reference_answers": [],
            "reference_contexts": [],
            "reference_context_ids": [],
            "reference_context_page_ids": [],
            "is_answerable_label": True,
        },
        default_template="templates.rag.end_to_end.json_predictions",
    ),
    "tasks.rag.end_to_end",
    overwrite=True,
)


add_to_catalog(
    Task(
        input_fields={
            "document_id": str,
            "title": str,
            "passages": List[Union[str, Image]],
            "metadata_tags": Dict[str, str],
        },
        reference_fields={},
        prediction_type=Any,
        metrics=[
            "metrics.rouge"
        ],  # We can not define an empty metric, so we gave here a simple one- although rouge is not related
        defaults={
            "title": "",
            "metadata_tags": {},
        },
    ),
    "tasks.rag.corpora",
    overwrite=True,
)
