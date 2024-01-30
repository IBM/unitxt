from unitxt.metrics import RemoteMetric
from unitxt.test_utils.metrics import test_metric

host = "127.0.0.1:8000"
endpoint = "http" + "://" + f"{host}/compute"
metric_name = "metrics.bert_score.deberta.xlarge.mnli"

metric = RemoteMetric(endpoint=endpoint, metric_name=metric_name)

bert_score_predictions = ["hello there general dude", "foo bar foobar"]
bert_score_references = [
    ["hello there general kenobi", "hello there!"],
    ["foo bar foobar", "foo bar"],
]
bert_score_instance_targets = [
    {"f1": 0.8, "precision": 0.86, "recall": 0.84, "score": 0.8, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
]

bert_score_global_target = {
    "f1": 0.9,
    "f1_ci_high": 1.0,
    "f1_ci_low": 0.8,
    "precision": 0.93,
    "precision_ci_high": 1.0,
    "precision_ci_low": 0.86,
    "recall": 0.92,
    "recall_ci_high": 1.0,
    "recall_ci_low": 0.84,
    "score": 0.9,
    "score_ci_high": 1.0,
    "score_ci_low": 0.8,
    "score_name": "f1",
}

test_metric(
    metric=metric,
    predictions=bert_score_predictions,
    references=bert_score_references,
    instance_targets=bert_score_instance_targets,
    global_target=bert_score_global_target,
)
