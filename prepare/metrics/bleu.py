from unitxt import add_to_catalog
from unitxt.metrics import HuggingfaceMetric
from unitxt.test_utils.metrics import test_metric

metric = HuggingfaceMetric(
    hf_metric_name="bleu", main_score="bleu", scale=1.0, prediction_type="str"
)

predictions = ["hello there general kenobi", "foo bar foobar", "", "not empty"]
references = [
    ["hello there general kenobi", "hello there !"],
    ["foo bar foobar"],
    ["not empty"],
    [""],
]

instance_targets = [
    {
        "bleu": 1.0,
        "precisions": [1.0, 1.0, 1.0, 1.0],
        "brevity_penalty": 1.0,
        "length_ratio": 1.33,
        "translation_length": 4,
        "reference_length": 3,
        "score": 1.0,
        "score_name": "bleu",
    },
    {
        "bleu": 0.0,
        "precisions": [1.0, 1.0, 1.0, 0.0],
        "brevity_penalty": 1.0,
        "length_ratio": 1.0,
        "translation_length": 3,
        "reference_length": 3,
        "score": 0.0,
        "score_name": "bleu",
    },
    {"score": None, "bleu": None, "score_name": "bleu"},
    {"score": None, "bleu": None, "score_name": "bleu"},
]

global_target = {
    "bleu": 0.9,
    "precisions": [0.78, 0.83, 1.0, 1.0],
    "brevity_penalty": 1.0,
    "length_ratio": 1.12,
    "translation_length": 9,
    "reference_length": 8,
    "score": 0.9,
    "score_name": "bleu",
    "bleu_ci_low": 0.9,
    "bleu_ci_high": 0.91,
    "score_ci_low": 0.9,
    "score_ci_high": 0.91,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.bleu", overwrite=True)
