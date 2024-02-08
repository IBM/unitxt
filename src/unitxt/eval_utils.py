from typing import List

import pandas as pd

from service.metrics.client_config import (
    get_remote_metrics_endpoint,
    get_remote_metrics_names,
)

from .artifact import verbosed_fetch_artifact
from .operator import SequentialOperator
from .operators import ApplyMetric
from .stream import MultiStream


def evaluate(dataset: pd.DataFrame, metric_names: List[str]):
    result = dataset.copy()
    remote_metrics = get_remote_metrics_names()
    for metric_name in metric_names:
        # prepare the input stream
        multi_stream = MultiStream.from_iterables(
            {"test": dataset.to_dict("records")}, copying=True
        )
        if metric_name in remote_metrics:
            metric = verbosed_fetch_artifact(metric_name)
            metric = ApplyMetric.as_remote_metric(metric)
            metric_step = metric
        else:
            # The SequentialOperator below will handle the load of the metric fromm its name
            metric_step = metric_name

        metrics_operator = SequentialOperator(steps=[metric_step])
        instances = list(metrics_operator(multi_stream)["test"])
        result[metric_name] = [
            instance["score"]["instance"]["score"] for instance in instances
        ]
    return result


def as_remote_metric(metric):
    from .metrics import MetricPipeline, RemoteMetric

    remote_metrics_endpoint = get_remote_metrics_endpoint()
    if isinstance(metric, MetricPipeline):
        metric = RemoteMetric.wrap_inner_metric_pipeline_metric(
            metric_pipeline=metric,
            remote_metrics_endpoint=remote_metrics_endpoint,
        )
    else:
        raise ValueError(
            f"Unexpected remote metric type {type(metric)} for the metric named '{metric.artifact_identifier}'. "
            f"Remotely executed metrics should be MetricPipeline objects."
        )
    return metric
