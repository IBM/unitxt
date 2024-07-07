from functools import singledispatch
from typing import List, Optional

import pandas as pd

from .artifact import verbosed_fetch_artifact
from .metric_utils import get_remote_metrics_endpoint, get_remote_metrics_names
from .operator import SequentialOperator
from .stream import MultiStream


@singledispatch
def evaluate(
    dataset, metric_names: List[str], compute_conf_intervals: Optional[bool] = False
):
    """Placeholder for overloading the function, supporting both dataframe input and list input."""
    pass


@evaluate.register
def _(
    dataset: list,
    metric_names: List[str],
    compute_conf_intervals: Optional[bool] = False,
):
    global_scores = {}
    remote_metrics = get_remote_metrics_names()
    for metric_name in metric_names:
        if metric_name in remote_metrics:
            metric = verbosed_fetch_artifact(metric_name)
            metric_step = as_remote_metric(metric)
        else:
            # The SequentialOperator below will handle the load of the metric from its name
            metric_step = metric_name
        metrics_operator = SequentialOperator(steps=[metric_step])

        if not compute_conf_intervals:
            first_step = metrics_operator.steps[0]
            first_step.disable_confidence_interval_calculation()

        multi_stream = MultiStream.from_iterables({"test": dataset}, copying=True)
        instances = list(metrics_operator(multi_stream)["test"])
        for entry, instance in zip(dataset, instances):
            entry[metric_name] = instance["score"]["instance"]["score"]

        if len(instances) > 0:
            global_scores[metric_name] = instances[0]["score"].get("global", {})

    return dataset, global_scores


@evaluate.register
def _(
    dataset: pd.DataFrame,
    metric_names: List[str],
    compute_conf_intervals: Optional[bool] = False,
):
    results, global_scores = evaluate(
        dataset.to_dict("records"),
        metric_names=metric_names,
        compute_conf_intervals=compute_conf_intervals,
    )
    return pd.DataFrame(results), pd.DataFrame(global_scores)


def as_remote_metric(metric):
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
            f"Unexpected remote metric type {type(metric)} for the metric named '{metric.__id__}'. "
            f"Remotely executed metrics should be MetricPipeline objects."
        )
    return metric
