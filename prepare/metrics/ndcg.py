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
            failure_defaults={"prediction": 0.0},
            use_nested_query=True,
        ),
    ],
    metric=NDCG(),
)


predictions = ["1.0", " 2 ", "1.0"]
references = [["4"], ["0"], ["1.0"]]

instance_targets = [  # nDCG is undefined at instance level
    {"nDCG": None, "score": None, "score_name": "nDCG"},
    {"nDCG": None, "score": None, "score_name": "nDCG"},
    {"nDCG": None, "score": None, "score_name": "nDCG"},
]

global_target = {
    "nDCG": 0.61,
    "nDCG_ci_high": 0.91,
    "nDCG_ci_low": 0.0,
    "score": 0.61,
    "score_ci_high": 0.91,
    "score_ci_low": 0.0,
    "score_name": "nDCG",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.ndcg", overwrite=True)
