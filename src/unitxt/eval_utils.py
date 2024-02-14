from functools import singledispatch
from typing import List, Optional

import pandas as pd

from .metrics import MetricPipeline
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
    for metric_name in metric_names:
        multi_stream = MultiStream.from_iterables({"test": dataset}, copying=True)
        metrics_operator = SequentialOperator(steps=[metric_name])

        if not compute_conf_intervals:
            first_step = metrics_operator.steps[0]
            if isinstance(first_step, MetricPipeline):
                first_step.metric.n_resamples = None
            else:
                first_step.n_resamples = None

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
    return pd.DataFrame(results), global_scores


#
