from typing import List, Union

from unitxt import add_to_catalog
from unitxt.blocks import (
    Task,
)

add_to_catalog(
    Task(
        __description__="""This is a task corresponding to the response generation step of RAG pipeline.
It assumes the input for is a set of questions and already retrieved contexts (documents or passsages).
The model response answer is evaluated against a set of reference_answers and/or using referenceless metrics such as the faithfullness
of the model answer to the provided context.

This task is similar to 'task.qa.with_context' , but supports multiple contexts and is focused only on text.

For details of RAG see: https://www.unitxt.ai/en/latest/docs/rag_support.html.
""",
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
        prediction_type=str,
        defaults={"contexts_ids": [], "reference_answers": []},
    ),
    "tasks.rag.response_generation",
    overwrite=True,
)
