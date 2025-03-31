from unitxt import add_to_catalog
from unitxt.metrics import MRR
from unitxt.test_utils.metrics import test_metric

metric = MRR()

predictions = [["a", "b", "c", "d", "e", "f"], ["g", "r", "u"], ["a", "b"], [], ["c"]]

references = [
    [["c", "d"]],  # third hit
    [["g"]],  # first hit
    [[]],  # no hit
    [["a"]],  # no hit
    [["b"]],  # no hit
]

instance_targets = [
    {"mrr": 0.33, "score": 0.33, "score_name": "mrr"},
    {"mrr": 1.0, "score": 1.0, "score_name": "mrr"},
    {"mrr": 0, "score": 0, "score_name": "mrr"},
    {"mrr": 0, "score": 0, "score_name": "mrr"},
    {"mrr": 0, "score": 0, "score_name": "mrr"},
]

global_target = {
    "mrr": 0.27,
    "score": 0.27,
    "score_name": "mrr",
    "mrr_ci_low": 0.07,
    "mrr_ci_high": 0.8,
    "score_ci_low": 0.07,
    "score_ci_high": 0.8,
    "num_of_instances": 5,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

predictions = [[1, 2, 3, 4, 5, 6], [7, 10, 11], [1, 2], [], [3]]

references = [
    [[3, 4]],  # third hit
    [[7]],  # first hit
    [[]],  # no hit
    [[1]],  # no hit
    [[2]],  # no hit
]

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.mrr", overwrite=True)
