import numpy as np

from src.unitxt import add_to_catalog
from src.unitxt.blocks import CastFields, CopyFields
from src.unitxt.metrics import KendallTauMetric, MetricPipeline
from src.unitxt.test_utils.metrics import test_metric

metric = MetricPipeline(
    main_score="kendalltau_b",
    preprocess_steps=[
        CopyFields(
            field_to_field=[("references/0", "references")],
            use_query=True,
        ),
        CastFields(
            fields={"prediction": "float", "references": "float"},
            failure_defaults={"prediction": 0.0},
            use_nested_query=True,
        ),
    ],
    metric=KendallTauMetric(),
)

predictions = ["1.0", " 2.0", "1.0"]
references = [["-1.0"], ["1.0"], ["0.0"]]

instance_targets = [
    {
        "kendalltau_b": np.nan,
        "score": np.nan,
        "score_name": "kendalltau_b",
        "p_val": np.nan,
    },
    {
        "kendalltau_b": np.nan,
        "score": np.nan,
        "score_name": "kendalltau_b",
        "p_val": np.nan,
    },
    {
        "kendalltau_b": np.nan,
        "score": np.nan,
        "score_name": "kendalltau_b",
        "p_val": np.nan,
    },
]

global_target = {
    "kendalltau_b": 0.82,
    "score": 0.82,
    "p_val": 0.22,
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
