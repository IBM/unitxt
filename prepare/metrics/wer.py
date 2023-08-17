from src.unitxt import add_to_catalog
from src.unitxt.metrics import Wer
from src.unitxt.test_utils.metrics import test_metric

metric = Wer()

predictions = ["this is the prediction", "there is an other sample"]
references = [["this is the reference"], ["there is another sample"]]

instance_targets = [
    {"wer": 0.25, "score": 0.25},  # 1 errors: reokace 'prediction' with 'reference'
    {"wer": 0.5, "score": 0.5},  # 2 errors: remove 'an' and replace 'other' with 'another'
]

global_target = {"wer": 0.38, "score": 0.38}  # Should by 0.375, but package rounds the scores

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.wer", overwrite=True)
