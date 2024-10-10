from unitxt import add_to_catalog
from unitxt.metrics import PredictionLength
from unitxt.test_utils.metrics import test_metric

metric = PredictionLength()

predictions = ["aaa", "bb", "ccccccc"]
references = [[""], [""], [""]]
instance_targets = [
    {"prediction_length": [3], "score": [3], "score_name": "prediction_length"},
    {"prediction_length": [2], "score": [2], "score_name": "prediction_length"},
    {"prediction_length": [7], "score": [7], "score_name": "prediction_length"},
]
global_target = {
    "prediction_length": 4.0,
    "score": 4.0,
    "score_name": "prediction_length",
    "num_of_evaluated_instances": 3,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.prediction_length", overwrite=True)
