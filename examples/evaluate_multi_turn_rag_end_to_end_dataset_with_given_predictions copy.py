from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.task import Task
from unitxt.templates import JsonOutputTemplate

logger = get_logger()

#
contexts = [
    "Austin is the capital of Texas.",
    "Houston is in Texas",
    "Houston is the the largest city in the state but not the capital of it.",
]

# Set up question answer pairs in a dictionary
dataset = [
    {
        "question": "What is the capital of Texas?",
        "conversation_id": 0,
        "turn_id": 0,
        "reference_answers": ["Austin"],
        "reference_contexts": [contexts[0]],
        "reference_context_ids": [0],
        "is_answerable_label": True,
    },
    {
        "question": "Which is the the largest city in the state?",
        "conversation_id": 0,
        "turn_id": 1,
        "reference_answers": ["Houston"],
        "reference_contexts": [contexts[1], contexts[2]],
        "reference_context_ids": [1, 2],
        "is_answerable_label": True,
    },
    {
        "question": "How much is 2+2?",
        "conversation_id": 1,
        "turn_id": 0,
        "reference_answers": ["4"],
        "reference_contexts": [""],
        "reference_context_ids": [],
        "is_answerable_label": True,
    },
    {
        "question": "Multiply the answer by 5",
        "conversation_id": 1,
        "turn_id": 1,
        "reference_answers": ["20"],
        "reference_contexts": [""],
        "reference_context_ids": [],
        "is_answerable_label": True,
    },
]

predictions = [
    {
        "answer": "Houston",
        "contexts": [contexts[2]],
        "context_ids": [2],
        "is_answerable": True,
    },
    {
        "answer": "Houston",
        "contexts": [contexts[2]],
        "context_ids": [2],
        "is_answerable": True,
    },
    {
        "answer": "4",
        "contexts": [""],
        "context_ids": [],
        "is_answerable": True,
    },
    {
        "answer": "25",
        "contexts": [""],
        "context_ids": [],
        "is_answerable": True,
    },
]

# select recommended metrics according to your available resources.
metrics = [
    "metrics.rag.end_to_end.recommended.cpu_only.all",
    # "metrics.rag.end_to_end.recommended.small_llm.all",
    # "metrics.rag.end_to_end.recommended.llmaj_watsonx.all",
    # "metrics.rag.end_to_end.recommended.llmaj_rits.all"
    # "metrics.rag.end_to_end.recommended.llmaj_azure.all"
]

multi_turn_rag_task = Task(
    input_fields={
        "question": "Union[str, Dialog]",
        "conversation_id": "Any",
        "turn_id": "Any",
        "metadata_tags": "Dict[str, str]",
    },
    reference_fields={
        "reference_answers": "List[str]",
        "reference_contexts": "List[str]",
        "reference_context_ids": "Union[List[int], List[str]]",
        "is_answerable_label": "bool",
    },
    metrics=[
        "metrics.rag.end_to_end.answer_correctness",
        "metrics.rag.end_to_end.answer_faithfulness",
        "metrics.rag.end_to_end.answer_reward",
        "metrics.rag.end_to_end.context_correctness",
        "metrics.rag.end_to_end.context_relevance",
    ],
    prediction_type="RagResponse",
    augmentable_inputs=[
        "question",
    ],
    defaults={
        "metadata_tags": {},
        "reference_answers": [],
        "reference_contexts": [],
        "reference_context_ids": [],
        "is_answerable_label": True,
    },
)

template = JsonOutputTemplate(
    input_format="Conversation: {conversation_id} Turn: {turn_id} Question: {question}",
    output_fields={
        "reference_answers": "answer",
        "reference_contexts": "contexts",
        "reference_context_ids": "context_ids",
    },
    wrap_with_list_fields=[
        "reference_contexts",
        "reference_context_ids",
    ],
    postprocessors=[
        "processors.load_json_predictions",
    ],
)

dataset = create_dataset(
    task=multi_turn_rag_task,
    test_set=dataset,
    split="test",
    postprocessors=[],
    metrics=metrics,
    template=template,
    group_by=["conversation_id"],
)

results = evaluate(predictions, dataset)

# Print Results:

print("Global Results:")
print(results.global_scores.summary)

print("Instance Results:")
print(results.instance_scores.summary)

print("Group results:")
print(results.groups_scores.summary)
