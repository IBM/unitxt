from unitxt import add_to_catalog
from unitxt.metrics import KPA
from unitxt.test_utils.metrics import test_metric

predictions = ["yes", "yes", "none", "none", "yes"]
references = [["yes"], ["none"], ["yes"], ["yes"], ["yes"]]
task_data = [{"keypoint": i} for i in ["1", "1", "2", "2", "2"]]
kpa_metric = KPA()

global_target = {
    "f1_1": 0.67,
    "f1_2": 0.5,
    "f1_macro": 0.58,
    "recall_macro": 0.67,
    "precision_macro": 0.75,
    "in_classes_support": 1.0,
    "f1_micro": 0.57,
    "recall_micro": 0.5,
    "precision_micro": 0.67,
    "score": 0.57,
    "score_name": "f1_micro",
    "score_ci_low": 0.33,
    "score_ci_high": 0.75,
    "f1_micro_ci_low": 0.33,
    "f1_micro_ci_high": 0.75,
    "num_of_evaluated_instances": 5,
}

instance_target = [
    {
        "f1_1": 1.0,
        "f1_macro": 1.0,
        "recall_macro": 1.0,
        "precision_macro": 1.0,
        "in_classes_support": 1.0,
        "f1_micro": 1.0,
        "recall_micro": 1.0,
        "precision_micro": 1.0,
        "score": 1.0,
        "score_name": "f1_micro",
    },
    {
        "f1_macro": 0.0,
        "recall_macro": 0.0,
        "precision_macro": 0.0,
        "in_classes_support": 0.0,
        "f1_micro": 0.0,
        "recall_micro": 0.0,
        "precision_micro": 0.0,
        "score": 0.0,
        "score_name": "f1_micro",
    },
    {
        "f1_2": 0.0,
        "f1_macro": 0.0,
        "recall_macro": 0.0,
        "precision_macro": 0.0,
        "in_classes_support": 1.0,
        "f1_micro": 0.0,
        "recall_micro": 0.0,
        "precision_micro": 0.0,
        "score": 0.0,
        "score_name": "f1_micro",
    },
    {
        "f1_2": 0.0,
        "f1_macro": 0.0,
        "recall_macro": 0.0,
        "precision_macro": 0.0,
        "in_classes_support": 1.0,
        "f1_micro": 0.0,
        "recall_micro": 0.0,
        "precision_micro": 0.0,
        "score": 0.0,
        "score_name": "f1_micro",
    },
    {
        "f1_2": 1.0,
        "f1_macro": 1.0,
        "recall_macro": 1.0,
        "precision_macro": 1.0,
        "in_classes_support": 1.0,
        "f1_micro": 1.0,
        "recall_micro": 1.0,
        "precision_micro": 1.0,
        "score": 1.0,
        "score_name": "f1_micro",
    },
]
outputs = test_metric(
    metric=kpa_metric,
    predictions=predictions,
    references=references,
    task_data=task_data,
    instance_targets=instance_target,
    global_target=global_target,
)

add_to_catalog(kpa_metric, "metrics.kpa", overwrite=True)
