from unitxt import add_to_catalog
from unitxt.metrics import Rouge
from unitxt.test_utils.metrics import test_metric

metric = Rouge(
    __description__="""This is the classical NLP Rouge metric based on the RougeScorer library (https://github.com/google-research/google-research/tree/master/rouge).
It computes metrics several metrics (rouge1, rouge2, roughL, and rougeLsum) based lexical (word) overlap between the prediction and the ground truth references."
""",
    __tags__={"flags": ["reference-based-metric", "cpu-metric"]},
)
predictions = ["hello there", "general kenobi"]
references = [["hello", "there"], ["general kenobi", "general yoda"]]

instance_targets = [
    {
        "rouge1": 0.67,
        "rouge2": 0.0,
        "rougeL": 0.67,
        "rougeLsum": 0.67,
        "score": 0.67,
        "score_name": "rougeL",
    },
    {
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
        "rougeLsum": 1.0,
        "score": 1.0,
        "score_name": "rougeL",
    },
]

global_target = {
    "rouge1": 0.83,
    "rouge1_ci_high": 1.0,
    "rouge1_ci_low": 0.67,
    "rouge2": 0.5,
    "rouge2_ci_high": 1.0,
    "rouge2_ci_low": 0.0,
    "rougeL": 0.83,
    "rougeL_ci_high": 1.0,
    "rougeL_ci_low": 0.67,
    "rougeLsum": 0.83,
    "rougeLsum_ci_high": 1.0,
    "rougeLsum_ci_low": 0.67,
    "score": 0.83,
    "score_ci_high": 1.0,
    "score_ci_low": 0.67,
    "score_name": "rougeL",
    "num_of_instances": 2,
}
outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
add_to_catalog(metric, "metrics.rouge", overwrite=True)
metric = Rouge(
    __deprecated_msg__=" Use 'metrics.rouge' which also generate confidence intervals"
)

add_to_catalog(
    metric,
    "metrics.rouge_with_confidence_intervals",
    overwrite=True,
)
