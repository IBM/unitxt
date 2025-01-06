from unitxt import add_to_catalog
from unitxt.metrics import HuggingfaceMetric, MeteorFast
from unitxt.test_utils.metrics import test_metric

metric = MeteorFast(
    n_resamples=3,
    __description__="""METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a machine translation evaluation metric, which is calculated based on the harmonic mean of precision and recall, with recall weighted more than precision.

METEOR is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations. Unigrams can be matched based on their surface forms, stemmed forms, and meanings. Once all generalized unigram matches between the two strings have been found, METEOR computes a score for this matching using a combination of unigram-precision, unigram-recall, and a measure of fragmentation that is designed to directly capture how well-ordered the matched words in the machine translation are in relation to the reference.
""",
)

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
    "num_of_instances": 4,
}

# to match the setting to occur by testing on the global version, metric2, below

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

metric_hf = MeteorFast(
    n_resamples=3,
    __description__="""Huggingface version with bad confidence interval calculation of METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a machine translation evaluation metric, which is calculated based on the harmonic mean of precision and recall, with recall weighted more than precision.

METEOR is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations. Unigrams can be matched based on their surface forms, stemmed forms, and meanings. Once all generalized unigram matches between the two strings have been found, METEOR computes a score for this matching using a combination of unigram-precision, unigram-recall, and a measure of fragmentation that is designed to directly capture how well-ordered the matched words in the machine translation are in relation to the reference.
""",
)

outputs = test_metric(
    metric=metric_hf,
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
add_to_catalog(metric_hf, "metrics.meteor_hf", overwrite=True)
