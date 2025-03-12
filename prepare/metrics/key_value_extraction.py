from unitxt import add_to_catalog
from unitxt.metrics import KeyValueExtraction
from unitxt.test_utils.metrics import test_metric

metric = KeyValueExtraction(__description__ = """ Metric that evaluates key value pairs predictions (provided as dictionary)
with reference key value pairs (also provided as dictionary). By default uses an accuracy (exact match) between each for the fields.
Reports average accuracy for each key , as well as micro and macro averages across all keys.

By overriding the metric field, it is possible to use other metrics to compare corresponding key values.
For example, KeyValueExtraction(metric=token_overlap) will calculate token overlap between corresponding keys.

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
add_to_catalog(metric, "metrics.key_value_extraction", overwrite=True)

