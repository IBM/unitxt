import numpy as np

from src.unitxt import add_to_catalog
from src.unitxt.metrics import KendallTauMetric
from src.unitxt.test_utils.metrics import test_metric

metric = KendallTauMetric()

predictions = ["1.0", "2.0", "1.0"]
references = [["-1.0"], ["1.0"], ["0.0"]]

instance_targets = [
    {
        "kendalltau_b": np.nan,
        "score": np.nan,
        "score_name": "kendalltau_b",
    }
] * 3

global_target = {
    "kendalltau_b": 0.82,
    "score": 0.82,
    "kendalltau_p_val": 0.22,
    "score_name": "kendalltau_b",
    "kendalltau_b_ci_low": np.nan,
    "kendalltau_b_ci_high": np.nan,
    "score_ci_low": np.nan,
    "score_ci_high": np.nan,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.kendalltau_b", overwrite=True)
