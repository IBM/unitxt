from src.unitxt import add_to_catalog
from src.unitxt.metrics import CharEditDistanceAccuracy
from src.unitxt.test_utils.metrics import test_metric

metric = CharEditDistanceAccuracy()

predictions = ["this is the prediction", "there is an other sample"]
references = [["this is the reference"], ["there is another sample"]]

# First sample:   p[re]diction - edit distance (8), len(21)  accuracy = 1 - 8/22 = 0.636
# Second samlple: [a]nother - edit distance (1), len(22)     accuracy = 1 - 1/22 = 0.958
instance_targets = [
    {"char_edit_dist_accuracy": 0.64, "score": 0.64},
    {"char_edit_dist_accuracy": 0.96, "score": 0.96},
]

global_target = {"char_edit_dist_accuracy": 0.8, "score": 0.8}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.char_edit_dist_accuracy", overwrite=True)
