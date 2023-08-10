from src.unitxt import add_to_catalog
from src.unitxt.metrics import Bleu
from src.unitxt.test_utils.metrics import test_metric

metric = Bleu()

predictions = ["hello there general kenobi", "foo bar foobar", "", "not empty"]
references = [["hello there general kenobi", "hello there !"], ["foo bar foobar"], ["not empty"], [""]]

instance_targets = [
    {
        "bleu": 1.0,
        "precisions": [1.0, 1.0, 1.0, 1.0],
        "brevity_penalty": 1.0,
        "length_ratio": 1.33,
        "translation_length": 4,
        "reference_length": 3,
        "score": 1.0,
    },
    {
        "bleu": 0.0,
        "precisions": [1.0, 1.0, 1.0, 0.0],
        "brevity_penalty": 1.0,
        "length_ratio": 1.0,
        "translation_length": 3,
        "reference_length": 3,
        "score": 0.0,
    },
    {"score": None, "bleu": None},
    {"score": None, "bleu": None},
]

global_target = {
    "bleu": 0.9,
    "precisions": [0.78, 0.83, 1.0, 1.0],
    "brevity_penalty": 1.0,
    "length_ratio": 1.12,
    "translation_length": 9,
    "reference_length": 8,
    "score": 0.9,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.bleu", overwrite=True)
