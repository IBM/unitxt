import numpy as np
from src.unitxt import add_to_catalog
from src.unitxt.metrics import MatthewsCorrelation
from src.unitxt.test_utils.metrics import test_metric

metric = MatthewsCorrelation()

predictions = ["A", "B", "A"]
references = [["A"], ["B"], ["B"]]

# a correlation score. single instance is always zero
instance_targets = [
    {"matthews_correlation": 0.0, "score": 0.0, "score_name": "matthews_correlation"},
    {"matthews_correlation": 0.0, "score": 0.0, "score_name": "matthews_correlation"},
    {"matthews_correlation": 0.0, "score": 0.0, "score_name": "matthews_correlation"},
]

global_target = {
    "matthews_correlation": 0.5,
    "score": 0.5,
    "score_name": "matthews_correlation",
    "matthews_correlation_ci_low": 0.0,
    "matthews_correlation_ci_high": 0.5,
    "score_ci_low": 0.0,
    "score_ci_high": 0.5,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.matthews_correlation", overwrite=True)
