import numpy as np
from unitxt import add_to_catalog
from unitxt.metrics import Pearsonr
from unitxt.test_utils.metrics import test_metric

metric = Pearsonr(n_resamples=100)
predictions = [1.0, 3.0, 1.1, 2.0, 8.0]
references = [[-1.0], [1.0], [0.10], [2.0], [6.0]]

instance_targets = [
    {"pearsonr": np.nan, "score": np.nan, "score_name": "pearsonr"},
    {"pearsonr": np.nan, "score": np.nan, "score_name": "pearsonr"},
    {"pearsonr": np.nan, "score": np.nan, "score_name": "pearsonr"},
    {"pearsonr": np.nan, "score": np.nan, "score_name": "pearsonr"},
    {"pearsonr": np.nan, "score": np.nan, "score_name": "pearsonr"},
]

expected_corr = 0.9202186222468902


global_target = {
    "num_of_instances": 5,
    "score": 0.95,
    "score_ci_high": 1.0,
    "score_ci_low": 0.3,
    "score_name": "pearsonr",
    "pearsonr": 0.95,
    "pearsonr_ci_high": 1.0,
    "pearsonr_ci_low": 0.3,
    "pearsonr_p_value": 0.01,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.pearson", overwrite=True)
