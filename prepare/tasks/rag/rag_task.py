from unitxt.blocks import (
    Task,
)
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={
            "question": "str",
            "question_id": "Any",
            "metadata_field": "str",
        },
        outputs={
            "reference_answers": "list[str]",
            "reference_contexts": "list[str]",
            "reference_context_ids": "list[str|int]",
            "is_answerable_label": "bool",
        },
        metrics=[
            "metrics.rag.answer_correctness",
            "metrics.rag.faithfulness",
            "metrics.rag.answer_reward",
            "metrics.rag.context_correctness",
            "metrics.rag.context_relevance",
        ],
        prediction_type="dict",
        augmentable_inputs=["question"],
    ),
    name="tasks.rag.end_to_end",
    overwrite=True,
)


add_to_catalog(
    Task(
        inputs={
            "document_id": "str",
            "title": "str",
            "passages": "List[str]",
            "metadata_field": "str",
        },
        outputs=[],
        metrics=["metrics.rouge"],
    ),
    name="tasks.rag.corpora",
    overwrite=True,
)
