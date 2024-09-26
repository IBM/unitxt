from unitxt import add_to_catalog
from unitxt.metrics import ANLS
from unitxt.test_utils.metrics import test_metric

metric = ANLS()

predictions = ["A", "B", "C"]
references = [["B"], ["A"], ["C"]]

instance_targets = [
    {"anls": 0.0, "score": 0.0, "score_name": "anls"},
    {"anls": 0.0, "score": 0.0, "score_name": "anls"},
    {"anls": 1.0, "score": 1.0, "score_name": "anls"},
]

global_target = {
    "anls": 0.33,
    "score": 0.33,
    "score_name": "anls",
    # "anls_ci_low": 0.0,
    # "anls_ci_high": 1.0,
    # "score_ci_low": 0.0,
    # "score_ci_high": 1.0,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.anls", overwrite=True)

