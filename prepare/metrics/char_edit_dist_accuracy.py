from unitxt import add_to_catalog
from unitxt.metrics import CharEditDistance, CharEditDistanceAccuracy
from unitxt.test_utils.metrics import test_metric

metric = CharEditDistanceAccuracy()
abs_dist_metric = CharEditDistance()

predictions = ["this is the prediction", "there is an other sample"]
references = [["this is the reference"], ["there is another sample"]]

# First sample:   p[re]diction - edit distance (8), max len ignoring whitespace (19)  accuracy = 1 - 8/19 = 0.578
# Second sample: [an other] [another] - edit distance ignoring white space(0), max len ignoring whitespace (19)     accuracy = 1 - 0/19 = 1

instance_targets = [
    {
        "char_edit_dist_accuracy": 0.58,
        "score": 0.58,
        "score_name": "char_edit_dist_accuracy",
    },
    {
        "char_edit_dist_accuracy": 1.00,
        "score": 1.00,
        "score_name": "char_edit_dist_accuracy",
    },
]

global_target = {
    "char_edit_dist_accuracy": 0.79,
    "score": 0.79,
    "score_name": "char_edit_dist_accuracy",
    "char_edit_dist_accuracy_ci_low": 0.58,
    "char_edit_dist_accuracy_ci_high": 1.0,
    "score_ci_low": 0.58,
    "score_ci_high": 1.0,
    "num_of_evaluated_instances": 2,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

dist_instance_targets = [
    {
        "char_edit_distance": 8,
        "score": 8,
        "score_name": "char_edit_distance",
    },
    {
        "char_edit_distance": 0,
        "score": 0,
        "score_name": "char_edit_distance",
    },
]

dist_global_target = {
    "char_edit_distance": 4.0,
    "score": 4.0,
    "score_name": "char_edit_distance",
    "char_edit_distance_ci_low": 0.0,
    "char_edit_distance_ci_high": 8.0,
    "score_ci_low": 0.0,
    "score_ci_high": 8.0,
    "num_of_evaluated_instances": 2,
}

test_metric(
    metric=abs_dist_metric,
    predictions=predictions,
    references=references,
    instance_targets=dist_instance_targets,
    global_target=dist_global_target,
)

predictions = [""]
references = [[""]]

instance_targets = [
    {
        "char_edit_dist_accuracy": 0.0,
        "score": 0.0,
        "score_name": "char_edit_dist_accuracy",
    }
]

global_target = {
    "char_edit_dist_accuracy": 0.0,
    "score": 0.0,
    "score_name": "char_edit_dist_accuracy",
    "num_of_evaluated_instances": 1,
}

test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

dist_instance_targets = [
    {
        "char_edit_distance": 0.0,
        "score": 0.0,
        "score_name": "char_edit_distance",
    }
]

dist_global_target = {
    "char_edit_distance": 0.0,
    "score": 0.0,
    "score_name": "char_edit_distance",
    "num_of_evaluated_instances": 1,
}

test_metric(
    metric=abs_dist_metric,
    predictions=predictions,
    references=references,
    instance_targets=dist_instance_targets,
    global_target=dist_global_target,
)


add_to_catalog(metric, "metrics.char_edit_dist_accuracy", overwrite=True)

add_to_catalog(abs_dist_metric, "metrics.char_edit_distance", overwrite=True)
