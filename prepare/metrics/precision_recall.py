from unitxt import add_to_catalog
from unitxt.metrics import BinaryPrecision, BinaryRecall
from unitxt.test_utils.metrics import test_metric

predictions = ["yes", "yes", "no"]
references = [["yes"], ["no"], ["yes"]]

precision_metric = BinaryPrecision()

instance_targets_precision = [
    {"binary_precision": 1.0, "score": 1.0, "score_name": "binary_precision"},
    {"binary_precision": 0.0, "score": 0.0, "score_name": "binary_precision"},
    {"binary_precision": 0.0, "score": 0.0, "score_name": "binary_precision"},
]

global_target_precision = {
    "binary_precision": 0.5,
    "score": 0.5,
    "score_name": "binary_precision",
    "score_ci_low": 0.02,
    "score_ci_high": 0.98,
    "binary_precision_ci_low": 0.02,
    "binary_precision_ci_high": 0.98,
}

outputs = test_metric(
    metric=precision_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_precision,
    global_target=global_target_precision,
)
add_to_catalog(precision_metric, "metrics.binary_precision", overwrite=True)


recall_metric = BinaryRecall()

global_target_recall = {
    "binary_recall": 0.5,
    "score": 0.5,
    "score_name": "binary_recall",
    "score_ci_low": 0.02,
    "score_ci_high": 0.66,
    "binary_recall_ci_low": 0.02,
    "binary_recall_ci_high": 0.66,
}

instance_targets_recall = [
    {"binary_recall": 1.0, "score": 1.0, "score_name": "binary_recall"},
    {"binary_recall": 0.0, "score": 0.0, "score_name": "binary_recall"},
    {"binary_recall": 0.0, "score": 0.0, "score_name": "binary_recall"},
]

outputs = test_metric(
    metric=recall_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_recall,
    global_target=global_target_recall,
)
add_to_catalog(recall_metric, "metrics.binary_recall", overwrite=True)
