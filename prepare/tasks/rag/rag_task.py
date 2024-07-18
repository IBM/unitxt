from dataclasses import dataclass

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog


@dataclass(frozen=True)
class TaskRagEndToEndConstants:
    # Templates
    TEMPLATE_RAG_END_TO_END_JSON_PREDICTIONS: str = (
        "templates.rag.end_to_end.json_predictions"
    )
    # Task names
    TASKS_RAG_END_TO_END: str = "tasks.rag.end_to_end"
    # Metrics

    METRICS_RAG_END_TO_END_ANSWER_CORRECTNESS: str = (
        "metrics.rag.end_to_end.answer_correctness"
    )
    METRICS_RAG_END_TO_END_ANSWER_REWARD: str = "metrics.rag.end_to_end.answer_reward"
    METRICS_RAG_END_TO_END_ANSWER_FAITHFULNESS: str = (
        "metrics.rag.end_to_end.answer_faithfulness"
    )
    METRICS_RAG_END_TO_END_CONTEXT_CORRECTNESS: str = (
        "metrics.rag.end_to_end.context_correctness"
    )
    METRICS_RAG_END_TO_END_CONTEXT_RELEVANCE: str = (
        "metrics.rag.end_to_end.context_relevance"
    )


@dataclass(frozen=True)
class TaskRagEndToEndInputConstants:
    QUESTION: str = "question"  # str
    QUESTION_ID: str = "question_id"  # str
    METADATA_FIELD: str = "metadata_field"  # str


@dataclass(frozen=True)
class TaskRagEndToEndReferenceConstants:
    REFERENCE_ANSWERS: str = "reference_answers"  # list[str]
    REFERENCE_CONTEXTS: str = "reference_contexts"  # list[str]
    REFERENCE_CONTEXT_IDS: str = "reference_context_ids"  # list[str|int]
    IS_ANSWERABLE_LABEL: str = "is_answerable_label"  # boolean


@dataclass(frozen=True)
class TaskRagEndToEndOutputConstants:
    QUESTION: str = "question"  # str
    PREDICTION: str = "prediction"  # str
    ANSWER: str = "answer"  # str
    CONTEXTS: str = "contexts"  # list[str]
    CONTEXT_IDS: str = "context_ids"  # list[str]


@dataclass(frozen=True)
class TaskRagCorporaInputConstants:
    DOCUMENT_ID: str = "document_id"  # str
    TITLE: str = "title"  # str
    PASSAGES: str = "passages"  # list[str]
    METADATA_FIELD: str = "metadata_field"  # str


@dataclass(frozen=True)
class TaskRagCorporaConstants:
    TASKS_RAG_CORPORA: str = "tasks.rag.corpora"


add_to_catalog(
    Task(
        inputs={
            TaskRagEndToEndInputConstants.QUESTION: "str",
            TaskRagEndToEndInputConstants.QUESTION_ID: "Any",
            TaskRagEndToEndInputConstants.METADATA_FIELD: "str",
        },
        outputs={
            TaskRagEndToEndReferenceConstants.REFERENCE_ANSWERS: "list[str]",
            TaskRagEndToEndReferenceConstants.REFERENCE_CONTEXTS: "list[str]",
            TaskRagEndToEndReferenceConstants.REFERENCE_CONTEXT_IDS: "list[str|int]",
            TaskRagEndToEndReferenceConstants.IS_ANSWERABLE_LABEL: "bool",
        },
        metrics=[
            TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_ANSWER_CORRECTNESS,
            TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_ANSWER_FAITHFULNESS,
            TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_ANSWER_REWARD,
            TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_CONTEXT_CORRECTNESS,
            TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_CONTEXT_RELEVANCE,
        ],
        prediction_type="dict",
        augmentable_inputs=[TaskRagEndToEndInputConstants.QUESTION],
    ),
    f"{TaskRagEndToEndConstants.TASKS_RAG_END_TO_END}",
)

add_to_catalog(
    Task(
        inputs={
            TaskRagCorporaInputConstants.DOCUMENT_ID: "str",
            TaskRagCorporaInputConstants.TITLE: "str",
            TaskRagCorporaInputConstants.PASSAGES: "List[str]",
            TaskRagCorporaInputConstants.METADATA_FIELD: "str",
        },
        outputs=[],
        metrics=["metrics.rouge"],
    ),
    f"{TaskRagCorporaConstants.TASKS_RAG_CORPORA}",
)
