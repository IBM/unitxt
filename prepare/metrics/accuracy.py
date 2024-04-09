from unitxt import add_to_catalog
from unitxt.metrics import Accuracy, BinaryAccuracy, BinaryMaxAccuracy
from unitxt.test_utils.metrics import test_metric

metric = Accuracy()

predictions = ["A", "B", "C"]
references = [["B"], ["A"], ["C"]]

instance_targets = [
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
]

global_target = {
    "accuracy": 0.33,
    "score": 0.33,
    "score_name": "accuracy",
    "accuracy_ci_low": 0.0,
    "accuracy_ci_high": 1.0,
    "score_ci_low": 0.0,
    "score_ci_high": 1.0,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.accuracy", overwrite=True)

metric = BinaryAccuracy()
add_to_catalog(metric, "metrics.accuracy_binary", overwrite=True)

metric = BinaryMaxAccuracy()
add_to_catalog(metric, "metrics.max_accuracy_binary", overwrite=True)
