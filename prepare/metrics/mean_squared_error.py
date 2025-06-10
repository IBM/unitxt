from unitxt import add_to_catalog
from unitxt.metrics import MeanSquaredError, RootMeanSquaredError
from unitxt.test_utils.metrics import test_metric

metric = MeanSquaredError(
    __description__="""Metric to calculate the mean squared error (MSE) between the prediction and the reference values.

    Assume both the prediction and reference are floats.

    Support only a single reference per prediction  .
    """
)
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


metric = RootMeanSquaredError(
    __description__="""Metric to calculate the root mean squared error (RMSE) between the prediction and the reference values.

    Assume both the prediction and reference are floats.

    Support only a single reference per prediction  .
    """
)


instance_targets = [
    {
        "root_mean_squared_error": 2.0,
        "score": 2.0,
        "score_name": "root_mean_squared_error",
    },
    {
        "root_mean_squared_error": 1.0,
        "score": 1.0,
        "score_name": "root_mean_squared_error",
    },
    {
        "root_mean_squared_error": 1.0,
        "score": 1.0,
        "score_name": "root_mean_squared_error",
    },
]

global_target = {
    "root_mean_squared_error": 1.41,
    "score": 1.41,
    "score_name": "root_mean_squared_error",
    "root_mean_squared_error_ci_low": 1.0,
    "root_mean_squared_error_ci_high": 2.0,
    "score_ci_low": 1.0,
    "score_ci_high": 2.0,
    "num_of_instances": 3,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.root_mean_squared_error", overwrite=True)
