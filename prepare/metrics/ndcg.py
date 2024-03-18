import numpy as np

from src.unitxt import add_to_catalog
from src.unitxt.blocks import CastFields, CopyFields
from src.unitxt.metrics import NDCG, MetricPipeline
from src.unitxt.test_utils.metrics import test_metric

# Normalized Discounted Cumulative Gain
metric = MetricPipeline(
    main_score="nDCG",
    preprocess_steps=[
        CopyFields(field_to_field=[("references/0", "references")], use_query=True),
        CastFields(
            fields={"prediction": "float", "references": "float"},
            failure_defaults={"prediction": None},
            use_nested_query=True,
        ),
    ],
    metric=NDCG(),
)

predictions = [
    "1.0",
    " 2 ",
    "1.0",
    "0",
    "1.7",
    3,
    "0",
    "oops",
    "1",
    "failed",
    "failed again",
]
references = [["4"], ["0"], ["1.0"], [4], ["0"], ["1"], ["1.0"], ["3"], ["2"], [4], [1]]
inputs = (
    [{"query": "who is Barack Obama"}] * 3
    + [{"query": "What is an albatross?"}] * 5
    + [{"query": "something else"}]
    + [{"query": "these will fail"}] * 2
)
instance_targets = [  # nDCG is undefined at instance level
    {"nDCG": np.nan, "score": np.nan, "score_name": "nDCG"}
] * len(predictions)

global_target = {
    "nDCG": 0.42,
    "nDCG_ci_high": 0.66,
    "nDCG_ci_low": 0.15,
    "score": 0.42,
    "score_ci_high": 0.66,
    "score_ci_low": 0.15,
    "score_name": "nDCG",
}
outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    task_data=inputs,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.ndcg", overwrite=True)
