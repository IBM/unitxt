from unitxt.blocks import (
    Task,
)
from unitxt.catalog import add_to_catalog

from prepare.tasks.rag.rag_constants import (
    TaskRagCorporaConstants,
    TaskRagCorporaInputConstants,
    TaskRagEndToEndConstants,
    TaskRagEndToEndInputConstants,
    TaskRagEndToEndReferenceConstants,
    TaskRagResponseGenConstants,
)

rag_response_generation_task = Task(
    inputs={
        TaskRagResponseGenConstants.CONTEXTS: "List[str]",
        TaskRagResponseGenConstants.CONTEXT_IDS: "List[int]",
        TaskRagResponseGenConstants.QUESTION: "str",
    },
    outputs={TaskRagResponseGenConstants.REFERENCE_ANSWERS: "List[str]"},
    metrics=[
        TaskRagResponseGenConstants.METRICS_ANSWER_CORRECTNESS,
        TaskRagResponseGenConstants.METRICS_ANSWER_FAITHFULNESS,
        TaskRagResponseGenConstants.METRICS_ANSWER_CORRECTNESS_BERT,
    ],
    augmentable_inputs=[
        TaskRagResponseGenConstants.CONTEXTS,
        TaskRagResponseGenConstants.QUESTION,
    ],
)

rag_end_to_end_task = Task(
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
        TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_ROUGE,
        TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_ANSWER_CORRECTNESS,
        TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_ANSWER_FAITHFULNESS,
        TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_ANSWER_REWARD,
        TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_CONTEXT_CORRECTNESS,
        TaskRagEndToEndConstants.METRICS_RAG_END_TO_END_CONTEXT_RELEVANCE,
    ],
    prediction_type="dict",
    augmentable_inputs=[TaskRagEndToEndInputConstants.QUESTION],
)

rag_corpora_task = Task(
    inputs={
        TaskRagCorporaInputConstants.DOCUMENT_ID: "str",
        TaskRagCorporaInputConstants.TITLE: "str",
        TaskRagCorporaInputConstants.PASSAGES: "List[str]",
        TaskRagCorporaInputConstants.METADATA_FIELD: "str",
    },
    outputs=[],
    metrics=["metrics.rouge"],
)

artifact_name = {
    rag_response_generation_task: TaskRagResponseGenConstants.TASKS_RAG_RESPONSE_GENERATION,
    rag_end_to_end_task: TaskRagEndToEndConstants.TASKS_RAG_END_TO_END,
    rag_corpora_task: TaskRagCorporaConstants.TASKS_RAG_CORPORA,
}

for artifact, name in artifact_name.items():
    add_to_catalog(artifact=artifact, name=name)
