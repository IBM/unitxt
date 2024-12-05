from unitxt import get_logger
from unitxt.api import create_and_evaluate_dataset
from unitxt.text_utils import print_dict

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
        "question_id": 0,
        "reference_answers": ["Austin"],
        "reference_contexts": [contexts[0]],
        "reference_context_ids": [0],
        "is_answerable_label": True,
    },
    {
        "question": "Which is the the largest city in Texas?",
        "question_id": 1,
        "reference_answers": ["Houston"],
        "reference_contexts": [contexts[1], contexts[2]],
        "reference_context_ids": [1, 2],
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
]

evaluated_dataset = create_and_evaluate_dataset(
    task="tasks.rag.end_to_end",
    predictions=predictions,
    data=dataset,
)

# Print results
for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "score",
        ],
    )
