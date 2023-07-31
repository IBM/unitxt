from src.unitxt import add_to_catalog
from src.unitxt.metrics import Bleu
from src.unitxt.test_utils.metrics import test_metric

metric = Bleu()

predictions = ["hello there general kenobi", "foo bar foobar"]
references = [["hello there general kenobi", "hello there !"], ["foo bar foobar"]]

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
]


global_target = {
    "bleu": 1.0,
    "precisions": [1.0, 1.0, 1.0, 1.0],
    "brevity_penalty": 1.0,
    "length_ratio": 1.17,
    "translation_length": 7,
    "reference_length": 6,
    "score": 1.0,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.bleu", overwrite=True)
