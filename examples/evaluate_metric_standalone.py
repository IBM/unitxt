import json

from unitxt.eval_utils import evaluate

references = [
    {"key1": "value1", "key2": "values2", "key3": "value3"},
    {"key1": "value3", "key2": "value4"},
]
predictions = [
    {"key1": "value1", "key2": "wrong-value", "wrong-key": "values3"},
    {"key1": "value3", "key2": "value4", "key3": "value9"},
]


# Use  from unitxt.eval_utils import evaluate
#
# Assumes the input is a list of dictionaries.
# Predictions and References are need to be set in respective fields ('prediction' and 'referencess')
# Note that 'references' must a list of possible references.  This is to support the case
# there are multiple correct references. If there is onnly one reference, it should be wrapped in a list.
#
# Returns a tuple:
#
# first element in the tuple is the original instance list , with the additional scores per instance (one per metric)
# [
#   {
#     "prediction": {
#       "key1": "value1",
#       "key2": "wrong-value",
#       "wrong-key": "values3"
#     },
#     "references": [
#       {
#         "key1": "value1",
#         "key2": "values2",
#         "key3": "value3"
#       }
#     ],
#     "metrics.key_value_extraction.token_overlap": 0.3333333333333333,
#     "metrics.key_value_extraction.accuracy": 0.3333333333333333
#   },
#   {
#     "prediction": {
#       "key1": "value3",
#       "key2": "value4",
#       "key3": "value9"
#     },
#     "references": [
#       {
#         "key1": "value3",
#         "key2": "value4"
#       }
#     ],
#     "metrics.key_value_extraction.token_overlap": 1.0,
#     "metrics.key_value_extraction.accuracy": 1.0
#   }
# ]
#
# The second tuple is a list of aggregated results per metric.
#
# Global_results:
# {
#   "metrics.key_value_extraction.token_overlap": {
#     "num_of_instances": 2,
#     "token_overlap_f1_key1": 1.0,
#     "token_overlap_f1_key2": 0.5,
#     "token_overlap_f1_key3": 0.0,
#     "token_overlap_f1_micro": 0.5,
#     "token_overlap_f1_macro": 0.5,
#     "token_overlap_f1_legal_keys_in_predictions": 0.8333333333333334,
#     "score": 0.5,
#     "score_name": "token_overlap_f1_micro"
#   },
#   "metrics.key_value_extraction.accuracy": {
#     "num_of_instances": 2,
#     "accuracy_key1": 1.0,
#     "accuracy_key2": 0.5,
#     "accuracy_key3": 0.0,
#     "accuracy_micro": 0.5,
#     "accuracy_macro": 0.5,
#     "accuracy_legal_keys_in_predictions": 0.8333333333333334,
#     "score": 0.5,
#     "score_name": "accuracy_micro"
#   }
# }


data = [
    {"prediction": prediction, "references": [reference]}
    for prediction, reference in zip(predictions, references)
]

instance_results, global_scores = evaluate(
    data,
    metric_names=[
        "metrics.key_value_extraction.token_overlap",
        "metrics.key_value_extraction.accuracy",
    ],
    compute_conf_intervals=False,
)

print("Instance results:")
print(json.dumps(instance_results, indent=2))

print("Global_results:")
print(json.dumps(global_scores, indent=2))
