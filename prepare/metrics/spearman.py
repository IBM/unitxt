import numpy as np
from src.unitxt import add_to_catalog
from src.unitxt.blocks import CastFields, CopyFields
from src.unitxt.metrics import HuggingfaceMetric, MetricPipeline
from src.unitxt.test_utils.metrics import test_metric

metric = MetricPipeline(
    main_score="spearmanr",
    preprocess_steps=[
        CopyFields(field_to_field=[("references/0", "references")], use_query=True),
        CastFields(
            fields={"prediction": "float", "references": "float"},
            failure_defaults={"prediction": 0.0},
            use_nested_query=True,
        ),
    ],
    metric=HuggingfaceMetric(
        hf_metric_name="spearmanr",
        main_score="spearmanr",
    ),
)

predictions = ["1.0", " 2.0", "1.0"]
references = [["-1.0"], ["1.0"], ["0.0"]]

instance_targets = [
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
    {"spearmanr": np.nan, "score": np.nan, "score_name": "spearmanr"},
]

global_target = {
    "spearmanr": 0.87,
    "score": 0.87,
    "score_name": "spearmanr",
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

add_to_catalog(metric, "metrics.spearman", overwrite=True)
