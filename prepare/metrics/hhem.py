from unitxt import add_to_catalog
from unitxt.metrics import FaithfulnessHHEM
from unitxt.test_utils.metrics import test_metric

pairs = [
    ("The capital of France is Berlin.", "The capital of France is Paris."),
    ("I am in California", "I am in United States."),
    ("I am in United States", "I am in California."),
]

predictions = [p[1] for p in pairs]
task_data = [{"contexts": [p[0]]} for p in pairs]

instance_targets = [
    {"score": 0.01, "score_name": "hhem_score", "hhem_score": 0.01},
    {"score": 0.65, "score_name": "hhem_score", "hhem_score": 0.65},
    {"score": 0.13, "score_name": "hhem_score", "hhem_score": 0.13},
]
global_target = {
    "num_of_instances": 3,
    "score": 0.26,
    "score_name": "hhem_score",
    "score_ci_low": 0.05,
    "score_ci_high": 0.65,
    "hhem_score": 0.26,
    "hhem_score_ci_low": 0.05,
    "hhem_score_ci_high": 0.65,
}

references = [[p[0]] for p in pairs]
metric = FaithfulnessHHEM()
outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
add_to_catalog(metric, "metrics.vectara_groundedness_hhem_2_1", overwrite=True)
