from unitxt import add_to_catalog
from unitxt.metrics import AccuracyFast, MultiTurnMetric, SequentialSuccess
from unitxt.test_utils.metrics import test_metric

metric = MultiTurnMetric(metric=AccuracyFast())

predictions = ["A", "B", "C"]
references = [["A"], ["A"], ["C"]]
task_data = [
    {
        "conversation": {
            "id": "aa",
            "dialog": [{"role": "user", "content": "what is it?"}],
        }
    },
    {
        "conversation": {
            "id": "aa",
            "dialog": [
                {"role": "user", "content": "what is it?"},
                {"role": "agent", "content": "A"},
                {"role": "user", "content": "what is it again?"},
            ],
        }
    },
    {
        "conversation": {
            "id": "bb",
            "dialog": [{"role": "user", "content": "what is it?"}],
        }
    },
]

instance_targets = [
    {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
]

global_target = {"accuracy": 0.75, "num_of_instances": 3, "score": 0.75, "score_name": "accuracy"}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    task_data=task_data,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.multi_turn.accuracy", overwrite=True)


metric = MultiTurnMetric(metric=AccuracyFast(), in_group_reduction=SequentialSuccess(), score_prefix="sequential_success_")


predictions = ["A", "B", "C"]
references = [["A"], ["A"], ["C"]]
task_data = [
    {
        "conversation": {
            "id": "aa",
            "dialog": [{"role": "user", "content": "what is it?"}],
        }
    },
    {
        "conversation": {
            "id": "aa",
            "dialog": [
                {"role": "user", "content": "what is it?"},
                {"role": "agent", "content": "A"},
                {"role": "user", "content": "what is it again?"},
            ],
        }
    },
    {
        "conversation": {
            "id": "bb",
            "dialog": [{"role": "user", "content": "what is it?"}],
        }
    },
]

instance_targets = [
    {"sequential_success_accuracy": 1.0, "score": 1.0, "score_name": "sequential_success_accuracy"},
    {"sequential_success_accuracy": 0.0, "score": 0.0, "score_name": "sequential_success_accuracy"},
    {"sequential_success_accuracy": 1.0, "score": 1.0, "score_name": "sequential_success_accuracy"},
]

global_target = {"num_of_instances": 3, "score": 0.75, "score_name": "sequential_success_accuracy", "sequential_success_accuracy": 0.75}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    task_data=task_data,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.multi_turn.sequential_success_accuracy", overwrite=True)
