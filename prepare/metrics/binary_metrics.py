from numpy import NaN

from src.unitxt import add_to_catalog
from src.unitxt.metrics import PrecisionMicroMultiLabel, RecallMicroMultiLabel
from src.unitxt.test_utils.metrics import test_metric

#
predictions = [["1"], ["1"], [], []]
references = [[["1"]], [[]], [["1"]], [["1"]]]

precision_metric = PrecisionMicroMultiLabel()


instance_targets_precision = [
    {"precision": 1.0, "score": 1.0, "score_name": "precision"},
    {"precision": NaN, "score": NaN, "score_name": "precision"},
    {"precision": 0.0, "score": 0.0, "score_name": "precision"},
    {"precision": 0.0, "score": 0.0, "score_name": "precision"},
]

global_target_precision = {
    "precision": 0.5,
    "score": 0.5,
    "score_name": "precision",
    "score_ci_low": 0.5,
    "score_ci_high": 0.86,
    "precision_ci_low": 0.5,
    "precision_ci_high": 0.86,
}

outputs = test_metric(
    metric=precision_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_precision,
    global_target=global_target_precision,
)
add_to_catalog(precision_metric, "metrics.precision", overwrite=True)


recall_metric = RecallMicroMultiLabel()

global_target_recall = {
    "recall": 0.33,
    "score": 0.33,
    "score_name": "recall",
    "score_ci_low": 0.33,
    "score_ci_high": 0.47,
    "recall_ci_low": 0.33,
    "recall_ci_high": 0.47,
}

instance_targets_recall = [
    {"recall": 1.0, "score": 1.0, "score_name": "recall"},
    {"recall": NaN, "score": NaN, "score_name": "recall"},
    {"recall": 0.0, "score": 0.0, "score_name": "recall"},
    {"recall": 0.0, "score": 0.0, "score_name": "recall"},
]
outputs = test_metric(
    metric=recall_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_recall,
    global_target=global_target_recall,
)
add_to_catalog(recall_metric, "metrics.recall", overwrite=True)
