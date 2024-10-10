from unitxt import add_to_catalog
from unitxt.metrics import MAP
from unitxt.test_utils.metrics import test_metric

metric = MAP()

predictions = [["a", "b", "c", "d", "e", "f"], ["g", "r", "u"], ["a", "b"], [], ["c"]]

references = [
    [["c", "d"]],  # third hit
    [["g"]],  # first hit
    [[]],  # no hit
    [["a"]],  # no hit
    [["b"]],
]  # no hit

instance_targets = [
    {"map": 0.42, "score": 0.42, "score_name": "map"},
    {"map": 1.0, "score": 1.0, "score_name": "map"},
    {"map": 0, "score": 0, "score_name": "map"},
    {"map": 0, "score": 0, "score_name": "map"},
    {"map": 0, "score": 0, "score_name": "map"},
]

global_target = {
    "map": 0.28,
    "score": 0.28,
    "score_name": "map",
    "map_ci_low": 0.0,
    "map_ci_high": 0.8,
    "score_ci_low": 0.0,
    "score_ci_high": 0.8,
    "num_of_evaluated_instances": 5,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.map", overwrite=True)
