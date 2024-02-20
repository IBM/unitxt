import numpy as np

from src.unitxt import add_to_catalog
from src.unitxt.metrics import RocAuc
from src.unitxt.test_utils.metrics import test_metric

metric = RocAuc()

predictions = ["0.2", "0.8", "1.0"]
references = [["1.0"], ["0.0"], ["1.0"]]

instance_targets = [{"roc_auc": np.nan, "score": np.nan, "score_name": "roc_auc"}] * 3
global_targets = {
    "roc_auc": 0.5,
    "roc_auc_ci_high": np.nan,
    "roc_auc_ci_low": np.nan,
    "score": 0.5,
    "score_ci_high": np.nan,
    "score_ci_low": np.nan,
    "score_name": "roc_auc",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_targets,
)

add_to_catalog(metric, "metrics.roc_auc", overwrite=True)
