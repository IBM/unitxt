import json
import os

UNITXT_REMOTE_METRICS_ENDPOINT = "UNITXT_REMOTE_METRICS_ENDPOINT"
UNITXT_REMOTE_METRICS = "UNITXT_REMOTE_METRICS"


def get_env_variable(variable_name: str, default_value: str) -> str:
    if variable_name not in os.environ:
        return default_value
    return os.environ[variable_name]


def get_metrics_client_config():
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
