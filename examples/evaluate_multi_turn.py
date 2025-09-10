import copy
from typing import List

from unitxt.api import create_dataset, evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.task import Task
from unitxt.templates import MultiTurnTemplate
from unitxt.types import Conversation

original_dialogs = [
    [
        {
            "role": "system",
            "content": "Have a dialog with the user and answer the questions.",
        },
        {"role": "user", "content": "Where is Paris?"},
        {"role": "assistant", "content": "Paris is in France"},
        {"role": "user", "content": "How is it also called?"},
        {"role": "assistant", "content": "City of Lights"},
    ],
    [
        {"role": "system", "content": "Calculate and return only the number."},
        {"role": "user", "content": "12+13"},
        {"role": "assistant", "content": "25"},
        {"role": "user", "content": "Multiply the result by 3.14159"},
        {"role": "assistant", "content": "78.53975"},
        {"role": "user", "content": "Multiply the result by 0"},
        {"role": "assistant", "content": "0"},
    ],
]

#
# Converts a flat list of dialog messages into an evaluation dataset where each assistant response is
# paired with the preceding conversation history as context.
#
# For every assistant message, the dialog up to the last user message is extracted and stored as a conversation,
# while the assistant's reply becomes the corresponding reference answer.
#
# For example, the following dialog
# [
#         {
#             "role": "system",
#             "content": "Have a dialog with the user and answer the questions.",
#         },
#         {"role": "user", "content": "Where is Paris?"},
#         {"role": "assistant", "content": "Paris is in France"},
#         {"role": "user", "content": "How is it also called?"},
#         {"role": "assistant", "content": "City of Lights"},
#     ]
#
# is converted to this evaluation set:
#
#  [
#     {
#         "conversation": {
#             "id": "1",
#             "dialog": [
#                 {
#                     "role": "system",
#                     "content": "Have a dialog with the user and answer the questions.",
#                 },
#                 {"role": "user", "content": "Where is Paris?"},
#             ],
#         },
#         "answers": ["Paris is in France"],
#     },
#     {
#         "conversation": {
#             "id": "1",
#             "dialog": [
#                 {
#                     "role": "system",
#                     "content": "Have a dialog with the user and answer the questions.",
#                 },
#                 {"role": "user", "content": "Where is Paris?"},
#                 {"role": "assistant", "content": "Paris is in France"},
#                 {"role": "user", "content": "How is it also called?"},
#             ],
#         },
#         "answers": ["The City of Lights"],
#     },
data = []
for id, dialog in enumerate(original_dialogs):
    new_dialog = []
    for turn in dialog:
        if turn["role"] == "assistant":
            new_data = {
                "conversation": {"dialog": copy.deepcopy(new_dialog), "id": str(id)},
                "answers": [turn["content"]],
            }
            data.append(new_data)
        new_dialog.append(turn)

template = MultiTurnTemplate(
    references_field="answers",
    turns_field="conversation/dialog",
)


criterion = "metrics.llm_as_judge.direct.criteria.correctness_based_on_ground_truth"
llm_as_judge_metric = f"metrics.llm_as_judge.direct.watsonx.llama3_3_70b[criteria={criterion}, context_fields=[answers]]"


task = Task(
    input_fields={
        "conversation": Conversation,
    },
    reference_fields={
        "answers": List[str],
    },
    prediction_type=str,
    metrics=[
        llm_as_judge_metric,
        "metrics.rouge",
        "metrics.normalized_sacrebleu",
        "metrics.accuracy",
    ],
    default_template=template,
)
model = CrossProviderInferenceEngine(model="llama-3-3-70b-instruct", provider="watsonx")

dataset = create_dataset(
    task=task,
    test_set=data,
    format="formats.chat_api",
    split="test",
    max_test_instances=10,
)
predictions = model.infer(dataset)

results = evaluate(predictions=predictions, data=dataset)

print("Instance Results:")
print(results.instance_scores.summary)

print("Global Results:")
print(results.global_scores.summary)
