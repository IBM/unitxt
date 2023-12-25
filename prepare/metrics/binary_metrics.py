from src.unitxt import add_to_catalog
from src.unitxt.metrics import BinaryF1, BinaryPrecision, BinaryRecall
from src.unitxt.test_utils.metrics import test_metric

predictions = ["1", "1", "0", "0"]
references = [["1"], ["0"], ["1"], ["1"]]

precision_metric = BinaryPrecision()


instance_targets_precision = [
    {"binary_precision": 1.0, "score": 1.0, "score_name": "binary_precision"},
    {"binary_precision": 0.0, "score": 0.0, "score_name": "binary_precision"},
    {"binary_precision": 0.0, "score": 0.0, "score_name": "binary_precision"},
    {"binary_precision": 0.0, "score": 0.0, "score_name": "binary_precision"},
]

global_target_precision = {
    "binary_precision": 0.5,
    "score": 0.5,
    "score_name": "binary_precision",
    "score_ci_low": 0.5,
    "score_ci_high": 0.86,
    "binary_precision_ci_low": 0.5,
    "binary_precision_ci_high": 0.86,
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
    "binary_recall": 0.33,
    "score": 0.33,
    "score_name": "binary_recall",
    "score_ci_low": 0.33,
    "score_ci_high": 0.47,
    "binary_recall_ci_low": 0.33,
    "binary_recall_ci_high": 0.47,
}

instance_targets_recall = [
    {"binary_recall": 1.0, "score": 1.0, "score_name": "binary_recall"},
    {"binary_recall": 0.0, "score": 0.0, "score_name": "binary_recall"},
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


f1_metric = BinaryF1()

global_target_recall = {
    "binary_f1": 0.4,
    "score": 0.4,
    "score_name": "binary_f1",
    "score_ci_low": 0.4,
    "score_ci_high": 0.62,
    "binary_f1_ci_low": 0.4,
    "binary_f1_ci_high": 0.62,
}

instance_targets_recall = [
    {"binary_f1": 1.0, "score": 1.0, "score_name": "binary_f1"},
    {"binary_f1": 0.0, "score": 0.0, "score_name": "binary_f1"},
    {"binary_f1": 0.0, "score": 0.0, "score_name": "binary_f1"},
    {"binary_f1": 0.0, "score": 0.0, "score_name": "binary_f1"},
]

outputs = test_metric(
    metric=f1_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_recall,
    global_target=global_target_recall,
)

add_to_catalog(f1_metric, "metrics.binary_f1", overwrite=True)
