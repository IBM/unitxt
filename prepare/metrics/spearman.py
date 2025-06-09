import numpy as np
from unitxt import add_to_catalog
from unitxt.metrics import Spearmanr
from unitxt.test_utils.metrics import test_metric

metric = Spearmanr(n_resamples=100)
predictions = [1.0, 3.0, 1.1, 2.0, 8.0]
references = [[-1.0], [1.0], [0.10], [2.0], [6.0]]

instance_targets = [
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
]

global_target = {
    "num_of_instances": 5,
    "score": 0.9,
    "score_ci_high": 1.0,
    "score_ci_low": 0.11,
    "score_name": "spearmanr",
    "spearmanr": 0.9,
    "spearmanr_ci_high": 1.0,
    "spearmanr_ci_low": 0.11,
    "spearmanr_p_value": 0.04,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.spearman", overwrite=True)
