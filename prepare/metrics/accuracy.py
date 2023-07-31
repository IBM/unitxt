from src.unitxt import add_to_catalog
from src.unitxt.metrics import Accuracy
from src.unitxt.test_utils.metrics import test_metric

metric = Accuracy()

predictions = ["A", "B", "C"]
references = [["B"], ["A"], ["C"]]

instance_targets = [{"accuracy": 0.0, "score": 0.0}, {"accuracy": 0.0, "score": 0.0}, {"accuracy": 1.0, "score": 1.0}]

global_target = {"accuracy": 0.33, "score": 0.33}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.accuracy", overwrite=True)
