from src.unitxt import add_to_catalog
from src.unitxt.metrics import RetrievalAtK
from src.unitxt.test_utils.metrics import test_metric

metric = RetrievalAtK(k_list=[1, 3, 5, 10, 20, 40])

predictions = [["a", "b", "c", "d", "e", "f"], ["g", "r", "u"], ["a", "b", "c"]]

references = [["c", "d"], ["g"], []]  # third hit  # first hit  # no hit

instance_targets = [
    {
        "match_at_1": 0.0,
        "match_at_3": 1.0,
        "match_at_5": 1.0,
        "match_at_10": 1.0,
        "match_at_20": 1.0,
        "match_at_40": 1.0,
        "precision_at_1": 0.0,
        "precision_at_3": 0.33,
        "precision_at_5": 0.4,
        "precision_at_10": 0.33,
        "precision_at_20": 0.33,
        "precision_at_40": 0.33,
        "recall_at_1": 0.0,
        "recall_at_3": 0.5,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "recall_at_20": 1.0,
        "recall_at_40": 1.0,
        "score": 0.0,
        "score_name": "match_at_1",
    },
    {
        "match_at_1": 1.0,
        "match_at_3": 1.0,
        "match_at_5": 1.0,
        "match_at_10": 1.0,
        "match_at_20": 1.0,
        "match_at_40": 1.0,
        "precision_at_1": 1.0,
        "precision_at_3": 0.33,
        "precision_at_5": 0.33,
        "precision_at_10": 0.33,
        "precision_at_20": 0.33,
        "precision_at_40": 0.33,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "recall_at_20": 1.0,
        "recall_at_40": 1.0,
        "score": 1.0,
        "score_name": "match_at_1",
    },
    {
        "match_at_1": 0.0,
        "match_at_3": 0.0,
        "match_at_5": 0.0,
        "match_at_10": 0.0,
        "match_at_20": 0.0,
        "match_at_40": 0.0,
        "precision_at_1": 0.0,
        "precision_at_3": 0.0,
        "precision_at_5": 0.0,
        "precision_at_10": 0.0,
        "precision_at_20": 0.0,
        "precision_at_40": 0.0,
        "recall_at_1": 0,
        "recall_at_3": 0,
        "recall_at_5": 0,
        "recall_at_10": 0,
        "recall_at_20": 0,
        "recall_at_40": 0,
        "score": 0.0,
        "score_name": "match_at_1",
    },
]

global_target = {
    "match_at_1": 0.33,
    "match_at_1_ci_high": 1.0,
    "match_at_1_ci_low": 0.0,
    "match_at_3": 0.67,
    "match_at_5": 0.67,
    "match_at_10": 0.67,
    "match_at_20": 0.67,
    "match_at_40": 0.67,
    "precision_at_1": 0.33,
    "precision_at_3": 0.22,
    "precision_at_5": 0.24,
    "precision_at_10": 0.22,
    "precision_at_20": 0.22,
    "precision_at_40": 0.22,
    "recall_at_1": 0.33,
    "recall_at_3": 0.5,
    "recall_at_5": 0.67,
    "recall_at_10": 0.67,
    "recall_at_20": 0.67,
    "recall_at_40": 0.67,
    "score": 0.33,
    "score_ci_high": 1.0,
    "score_ci_low": 0.0,
    "score_name": "match_at_1",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.retrieval_at_k", overwrite=True)
