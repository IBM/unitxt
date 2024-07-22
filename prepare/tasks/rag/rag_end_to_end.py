from unitxt import add_to_catalog
from unitxt.blocks import Task

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
            "metrics.rag.end_to_end.answer_correctness",
            "metrics.rag.end_to_end.answer_faithfulness",
            "metrics.rag.end_to_end.answer_reward",
            "metrics.rag.end_to_end.context_correctness",
            "metrics.rag.end_to_end.context_relevance",
        ],
        prediction_type="dict",
        augmentable_inputs=["question"],
    ),
    "tasks.rag.end_to_end",
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
    "tasks.rag.corpora",
    overwrite=True,
)
