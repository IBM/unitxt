from unitxt import add_to_catalog
from unitxt.metrics import ExactMatchMM
from unitxt.test_utils.metrics import test_metric

metric = ExactMatchMM(n_resamples=None)

predictions = ["A", "B", "C"]
references = [["B"], ["A"], ["C"]]

instance_targets = [
    {"exact_match_mm": 0.0, "score": 0.0, "score_name": "exact_match_mm"},
    {"exact_match_mm": 0.0, "score": 0.0, "score_name": "exact_match_mm"},
    {"exact_match_mm": 1.0, "score": 1.0, "score_name": "exact_match_mm"},
]

global_target = {
    "exact_match_mm": 0.33,
    "score": 0.33,
    "score_name": "exact_match_mm",
    "num_of_instances": 3,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.exact_match_mm", overwrite=True)
