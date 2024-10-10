from unitxt import add_to_catalog
from unitxt.metrics import HuggingfaceMetric, Meteor
from unitxt.test_utils.metrics import test_metric

metric = Meteor()

predictions = [
    "It is a guide to action which ensures that the military always obeys the commands of the party",
    "We strive for peace",
    "On the rag sat the cat",
    "I caught the ball",
]
references = [
    [
        "It is a guide to action that ensures that the military will forever heed Party commands"
    ],
    ["We hope for peace"],
    ["The cat sat on the rag"],
    ["He threw the ball"],
]

# the floats shown here are rounded just for the test. the actually
# returned score are 15-16 digits to the right of the decimal point
instance_targets = [
    {"meteor": 0.69, "score": 0.69, "score_name": "meteor"},
    {"meteor": 0.64, "score": 0.64, "score_name": "meteor"},
    {"meteor": 0.5, "score": 0.5, "score_name": "meteor"},
    {"meteor": 0.47, "score": 0.47, "score_name": "meteor"},
]

global_target = {
    "meteor": 0.58,
    "meteor_ci_high": 0.59,
    "meteor_ci_low": 0.58,
    "score": 0.58,
    "score_ci_high": 0.59,
    "score_ci_low": 0.58,
    "score_name": "meteor",
    "num_of_evaluated_instances": 4,
}

metric.n_resamples = 3
# to match the setting to occur by testing on the global version, metric2, below

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

# compare results with the HF version of meteor
metric2 = HuggingfaceMetric(
    hf_metric_name="meteor", main_score="meteor", prediction_type=str
)

outputs = test_metric(
    metric=metric2,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.meteor", overwrite=True)
