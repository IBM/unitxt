from unitxt import add_to_catalog
from unitxt.metrics import Rouge
from unitxt.test_utils.metrics import test_metric

metric = Rouge(n_resamples=None)

predictions = ["hello there", "general kenobi"]
references = [["hello", "there"], ["general kenobi", "general yoda"]]

instance_targets = [
    {
        "rouge1": 0.67,
        "rouge2": 0.0,
        "rougeL": 0.67,
        "rougeLsum": 0.67,
        "score": 0.67,
        "score_name": "rougeL",
    },
    {
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
        "rougeLsum": 1.0,
        "score": 1.0,
        "score_name": "rougeL",
    },
]

global_target = {
    "rouge1": 0.83,
    "rouge2": 0.5,
    "rougeL": 0.83,
    "rougeLsum": 0.83,
    "score": 0.83,
    "score_name": "rougeL",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
add_to_catalog(metric, "metrics.rouge", overwrite=True)

global_target_with_confidence_intervals = global_target.copy()
global_target_with_confidence_intervals.update(
    {
        "rougeL_ci_low": 0.83,
        "rougeL_ci_high": 0.83,
        "score_ci_low": 0.83,
        "score_ci_high": 0.83,
    }
)

metric_with_confidence_intervals = Rouge()
outputs = test_metric(
    metric=metric_with_confidence_intervals,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target_with_confidence_intervals,
)
add_to_catalog(
    metric_with_confidence_intervals,
    "metrics.rouge_with_confidence_intervals",
    overwrite=True,
)
