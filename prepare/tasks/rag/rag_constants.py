from dataclasses import dataclass


@dataclass(frozen=True)
class TaskRagEndToEndConstants:
    # Templates
    TEMPLATE_RAG_END_TO_END_JSON_PREDICTIONS: str = (
        "templates.rag.end_to_end.json_predictions"
    )
    # Task names
    TASKS_RAG_END_TO_END: str = "tasks.rag.end_to_end"

    # Metrics
    METRICS_RAG_END_TO_END_ROUGE: str = "metrics.rag..rouge"

    METRICS_RAG_END_TO_END_ANSWER_CORRECTNESS: str = "metrics.rag.answer_correctness"
    METRICS_RAG_END_TO_END_ANSWER_REWARD: str = "metrics.rag.end_to_end.answer_reward"
    METRICS_RAG_END_TO_END_ANSWER_FAITHFULNESS: str = "metrics.rag.answer_faithfulness"
    METRICS_RAG_END_TO_END_CONTEXT_CORRECTNESS: str = "metrics.rag.context_correctness"
    METRICS_RAG_END_TO_END_CONTEXT_RELEVANCE: str = "metrics.rag.context_relevance"


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


@dataclass(frozen=True)
class TaskRagResponseGenConstants:
    QUESTION: str = "question"  # str
    CONTEXTS: str = "contexts"  # list[str]
    CONTEXT_IDS: str = "context_ids"  # list[str]
    REFERENCE_ANSWERS: str = "reference_answers"  # list[str]

    METRICS_ANSWER_CORRECTNESS: str = (
        "metrics.rag.response_generation.correctness.token_overlap"
    )
    METRICS_ANSWER_CORRECTNESS_BERT: str = (
        "metrics.rag.response_generation.correctness.bert_score.deberta_large_mnli"
    )
    METRICS_ANSWER_FAITHFULNESS: str = (
        "metrics.rag.response_generation.faithfullness.token_overlap"
    )

    TASKS_RAG_RESPONSE_GENERATION: str = "tasks.rag.response_generation"
