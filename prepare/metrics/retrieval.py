from src.unitxt import add_to_catalog
from src.unitxt.metrics import RetrievalAtK
from src.unitxt.test_utils.metrics import test_metric

metric = RetrievalAtK(k_list=[1, 3])

predictions = [["a", "b", "c", "d", "e", "f"], ["g", "r", "u"], ["a", "b", "c"]]

references = [["c", "d"], ["g"], []]  # third hit  # first hit  # no hit

instance_targets = [
    {
        "match_at_1": 0.0,
        "match_at_3": 1.0,
        "precision_at_1": 0.0,
        "precision_at_3": 0.33,
        "recall_at_1": 0.0,
        "recall_at_3": 0.5,
        "score": 0.0,
        "score_name": "match_at_1",
    },
    {
        "match_at_1": 1.0,
        "match_at_3": 1.0,
        "precision_at_1": 1.0,
        "precision_at_3": 0.33,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "score": 1.0,
        "score_name": "match_at_1",
    },
    {
        "match_at_1": 0.0,
        "match_at_3": 0.0,
        "precision_at_1": 0.0,
        "precision_at_3": 0.0,
        "recall_at_1": 0,
        "recall_at_3": 0,
        "score": 0.0,
        "score_name": "match_at_1",
    },
]

global_target = {
    "score_ci_high": 1.0,
    "score_ci_low": 0.0,
    "score": 0.33,
    "score_name": "match_at_1",
    "match_at_1_ci_high": 1.0,
    "match_at_1_ci_low": 0.0,
    "match_at_1": 0.33,
    "match_at_3": 0.67,
    "precision_at_1": 0.33,
    "precision_at_3": 0.22,
    "recall_at_1": 0.33,
    "recall_at_3": 0.5,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.retrieval_at_k", overwrite=True)
