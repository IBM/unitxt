from functools import singledispatch
from typing import List

import pandas as pd

from .artifact import verbosed_fetch_artifact
from .metric import Metric
from .metric_utils import get_remote_metrics_endpoint, get_remote_metrics_names
from .operator import SequentialOperator
from .stream import MultiStream


@singledispatch
def evaluate(dataset, metric_names: List[str]):
    """Placeholder for overloading the function, supporting both dataframe input and list input."""
    pass


@evaluate.register
def _(dataset: list, metric_names: List[str]):
    remote_metrics = get_remote_metrics_names()
    for metric_name in metric_names:
        multi_stream = MultiStream.from_iterables({"test": dataset}, copying=True)
        if metric_name in remote_metrics:
            metric = verbosed_fetch_artifact(metric_name)
            metric_step = as_remote_metric(metric)
        else:
            # The SequentialOperator below will handle the load of the metric fromm its name
            metric_step = metric_name
        metrics_operator = SequentialOperator(steps=[metric_step])
        instances = list(metrics_operator(multi_stream)["test"])
        for entry, instance in zip(dataset, instances):
            entry[metric_name] = instance["score"]["instance"]["score"]
    return dataset


@evaluate.register
def _(dataset: pd.DataFrame, metric_names: List[str]):
    return pd.DataFrame(evaluate(dataset.to_dict("records"), metric_names=metric_names))


def as_remote_metric(metric: Metric) -> Metric:
    """Wrap a metric with a RemoteMetric.

    Currently supported is wrapping the inner metric within a MetricPipeline.
    """
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
