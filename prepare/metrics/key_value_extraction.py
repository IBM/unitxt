from unitxt import add_to_catalog
from unitxt.metrics import KeyValueExtraction
from unitxt.test_utils.metrics import test_metric

metric = KeyValueExtraction(__description__ = """Metric that evaluates key value pairs predictions (provided as dictionaries)
with reference key value pairs (also provided as dictionaries). By default uses an accuracy (exact match) between each for the fields.
Reports average accuracy for each key , as well as micro and macro averages across all keys.
""")

predictions = [
    {"key1": "value1", "key2": "value2", "unknown_key": "unknown_value"}
]

references = [[{"key1": "value1", "key2" : "value3"}]]
#
instance_targets = [
     {"accuracy_key1": 1.0, "accuracy_key2": 0.0, "accuracy_legal_keys_in_predictions": 0.67, "accuracy_macro": 0.5, "accuracy_micro": 0.5, "score": 0.5, "score_name": "accuracy_micro"}
]
global_target = {"accuracy_key1": 1.0, "accuracy_key2": 0.0, "accuracy_legal_keys_in_predictions": 0.67, "accuracy_macro": 0.5, "accuracy_micro": 0.5, "score": 0.5, "score_name": "accuracy_micro", "num_of_instances" : 1}
outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
add_to_catalog(metric, "metrics.key_value_extraction.accuracy", overwrite=True)

metric = KeyValueExtraction(__description__ = """Metric that evaluates key value pairs predictions (provided as dictionary)
with reference key value pairs (also provided as dictionary).
Calculates token overlap between values of corresponding value in reference and prediction.
Reports f1 per key ,  micro f1 averages across all key/value pairs, and macro f1 averages across keys.
""",
metric="token_overlap")

add_to_catalog(metric, "metrics.key_value_extraction.token_overlap", overwrite=True)
