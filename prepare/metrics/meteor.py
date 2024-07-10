from unitxt import add_to_catalog
from unitxt.metrics import Meteor
from unitxt.test_utils.metrics import test_metric

metric = Meteor()

predictions = [
    "It is a guide to action which ensures that the military always obeys the commands of the party",
    "We strive for peace",
]
references = [
    [
        "It is a guide to action that ensures that the military will forever heed Party commands"
    ],
    ["We hope for peace"],
]

# the floats shown here are rounded just for the test. the actually
# returned score are 15-16 digits to the right of the decimal point
instance_targets = [
    {"meteor": 0.69, "score": 0.69, "score_name": "meteor"},
    {"meteor": 0.64, "score": 0.64, "score_name": "meteor"},
]

global_target = {
    "meteor": 0.67,
    "meteor_ci_high": 0.69,
    "meteor_ci_low": 0.64,
    "score": 0.67,
    "score_ci_high": 0.69,
    "score_ci_low": 0.64,
    "score_name": "meteor",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.meteor", overwrite=True)
