from unitxt import add_to_catalog
from unitxt.metrics import MeanSquaredError
from unitxt.test_utils.metrics import test_metric

metric=MeanSquaredError()
predictions = [1.0, 2.0, 1.0]
references = [[-1.0], [1.0], [0.0]]

instance_targets = [
    {"mean_squared_error": 4.0, "score": 4.0, "score_name": "mean_squared_error"},
    {"mean_squared_error": 1.0, "score": 1.0, "score_name": "mean_squared_error"},
    {"mean_squared_error": 1.0, "score": 1.0, "score_name": "mean_squared_error"},
]

global_target = {
    "mean_squared_error": 2.0,
    "score": 2.0,
    "score_name": "mean_squared_error",
    "mean_squared_error_ci_low": 1.0,
    "mean_squared_error_ci_high": 4.0,
    "score_ci_low": 1.0,
    "score_ci_high": 4.0,
    "num_of_instances": 3,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.mean_squared_error", overwrite=True)
