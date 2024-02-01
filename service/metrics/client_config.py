import json
import os

"""
This module enables loading the metric service client configuration from
local environment variables
"""

# A list of metrics to be executed remotely.
# For example: '["metrics.rag.context_relevance","metrics.rag.bert_k_precision"]'
# This value should be a valid json list
UNITXT_REMOTE_METRICS = "UNITXT_REMOTE_METRICS"

# The remote endpoint on which the remote metrics are available.
# For example, 'http://127.0.0.1:8000/compute'
UNITXT_REMOTE_METRICS_ENDPOINT = "UNITXT_REMOTE_METRICS_ENDPOINT"


def get_env_variable(variable_name: str, default_value: str) -> str:
    if variable_name not in os.environ:
        return default_value
    return os.environ[variable_name]


def get_metrics_client_config():
    """Load the remote metrics configuration from environment variables.

    Returns:
        A tuple (remote_metrics, remote_metrics_endpoint) containing
            remote_metrics: List[str] - names of metrics to be executed remotely.
            remote_metrics_endpoint: str - The remote endpoint on which the remote metrics are available.
    """
    remote_metrics = get_env_variable(UNITXT_REMOTE_METRICS, default_value=[])
    remote_metrics = json.loads(remote_metrics)
    if not isinstance(remote_metrics, list):
        raise RuntimeError(
            f"Unexpected value {remote_metrics} for the '{UNITXT_REMOTE_METRICS}' environment variable. "
            f"The value is expected to be a list of metric names in json format."
        )
    for remote_metric in remote_metrics:
        if not isinstance(remote_metric, str):
            raise RuntimeError(
                f"Unexpected value {remote_metric} within the '{UNITXT_REMOTE_METRICS}' environment variable. "
                f"The value is expected to be a string but its type is {type(remote_metric)}."
            )
    # if remote_metrics is not an empty list, this feature is enabled,
    # and an endpoint should be defined in the environment variables.
    if remote_metrics:
        remote_metrics_endpoint = get_env_variable(
            UNITXT_REMOTE_METRICS_ENDPOINT, default_value=None
        )
        if not remote_metrics_endpoint:
            raise RuntimeError(
                f"Unexpected None value for '{UNITXT_REMOTE_METRICS_ENDPOINT}'. "
                f"Running metrics {remote_metrics} as remote metrics requires defining an "
                f"endpoint in the environment variable '{UNITXT_REMOTE_METRICS_ENDPOINT}'."
            )
    return remote_metrics, remote_metrics_endpoint
