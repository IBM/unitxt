import json
import os
from copy import deepcopy
from typing import List

import pandas as pd

from .artifact import verbosed_fetch_artifact
from .metrics import MetricPipeline, RemoteMetric
from .operator import SequentialOperator
from .stream import MultiStream


def get_env_variable(variable_name: str, default_value: str) -> str:
    if variable_name not in os.environ:
        return default_value
    return os.environ[variable_name]


UNITXT_REMOTE_METRICS_ENDPOINT = "UNITXT_REMOTE_METRICS_ENDPOINT"
UNITXT_REMOTE_METRICS = "UNITXT_REMOTE_METRICS"


def evaluate(dataset: pd.DataFrame, metric_names: List[str]):
    result = dataset.copy()
    remote_metrics = get_env_variable(UNITXT_REMOTE_METRICS, default_value=[])
    remote_metrics = json.loads(remote_metrics)
    if not isinstance(remote_metrics, list):
        raise ValueError(
            f"Unexpected value {remote_metrics} for the {UNITXT_REMOTE_METRICS} environment variable."
            f"The value is expected to be a list of metric names in json format."
        )
    if remote_metrics:
        remote_metrics_endpoint = get_env_variable(
            UNITXT_REMOTE_METRICS_ENDPOINT, default_value=None
        )
        if not remote_metrics_endpoint:
            raise RuntimeError(
                f"Unexpected None value for {UNITXT_REMOTE_METRICS_ENDPOINT}. "
                f"Running metrics {remote_metrics} as remote metrics requires defining an "
                f"endpoint in the environment variable '{UNITXT_REMOTE_METRICS_ENDPOINT}'."
            )
    # prepare the input stream
    for metric_name in metric_names:
        multi_stream = MultiStream.from_iterables(
            {"test": dataset.to_dict("records")}, copying=True
        )
        if metric_name in remote_metrics:
            metric = verbosed_fetch_artifact(metric_name)
            if isinstance(metric, MetricPipeline):
                inner_metric_identifier = metric.metric.artifact_identifier
                metric = deepcopy(metric)
                metric.metric = RemoteMetric(
                    main_score=metric_name,
                    metric_name=inner_metric_identifier,
                    endpoint=remote_metrics_endpoint,
                )
                metric_step = metric
            else:
                raise ValueError(
                    f"Unexpected remote metric type {type(metric)} for the metric named '{metric_name}'. "
                    f"Remotely executed metrics should be MetricPipeline objects."
                )
                # metric_step = RemoteMetric(main_score=metric_name, metric_name=metric_name, endpoint=remote_metrics_endpoint)
        else:
            metric_step = metric_name

        metrics_operator = SequentialOperator(steps=[metric_step])
        instances = list(metrics_operator(multi_stream)["test"])
        result[metric_name] = [
            instance["score"]["instance"]["score"] for instance in instances
        ]
    return result
