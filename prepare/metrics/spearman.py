import numpy as np
from unitxt import add_to_catalog
from unitxt.metrics import MetricPipeline, Spearmanr
from unitxt.operators import Copy
from unitxt.test_utils.metrics import test_metric

metric = MetricPipeline(
    main_score="spearmanr",
    preprocess_steps=[
        Copy(field="references/0", to_field="references"),
    ],
    metric=Spearmanr(),
    prediction_type=float,
)

predictions = [1.0, 2.0, 1.0]
references = [[-1.0], [1.0], [0.0]]

instance_targets = [
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
]

global_target = {
    "spearmanr": 0.87,
    "score": 0.87,
    "score_name": "spearmanr",
    "spearmanr_ci_low": np.nan,
    "spearmanr_ci_high": np.nan,
    "score_ci_low": np.nan,
    "score_ci_high": np.nan,
    "num_of_instances": 3,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.spearman", overwrite=True)
