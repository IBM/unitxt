from unitxt.test_utils.metrics import RemoteMetric, test_metric

from prepare.metrics.rag import (
    bert_score_global_target,
    bert_score_instance_targets,
    bert_score_predictions,
    bert_score_references,
)

host = "127.0.0.1:8000"
metric = "metrics.bert_score.deberta.xlarge.mnli"
metric_url = RemoteMetric(host=host, metric=metric)

assert test_metric(
    metric=metric_url,
    predictions=bert_score_predictions,
    references=bert_score_references,
    instance_targets=bert_score_instance_targets,
    global_target=bert_score_global_target,
)
