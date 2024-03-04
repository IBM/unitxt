import json
from typing import Any, Dict, Iterable, List, Optional

from datasets import Features, Value

from .dataclass import Dataclass
from .operator import (
    MultiStreamOperator,
    SequentialOperatorInitilizer,
    StreamInitializerOperator,
)
from .operators import (
    Apply,
    ApplyMetric,
    ApplyOperatorsField,
    FlattenInstances,
    MergeStreams,
    SplitByValue,
)
from .register import _reset_env_local_catalogs, register_all_artifacts
from .schema import UNITXT_DATASET_SCHEMA
from .settings_utils import get_settings
from .stream import MultiStream, Stream


class MultiStreamScoreMean(MultiStreamOperator):
    def aggegate_results(self, multi_stream: MultiStream):
        scores = []
        for stream in multi_stream.values():
            instance = stream.peek()
            scores.append(instance["score"]["global"]["score"])

        from statistics import mean

        return mean(scores)

    def spread_results(self, stream: Stream, score: float):
        for instance in stream:
            instance["score"]["global"]["groups_mean_score"] = score
            yield instance

    def spread_results_one_stream(self, stream: Stream):
        for instance in stream:
            instance["score"]["global"]["groups_mean_score"] = instance["score"][
                "global"
            ]["score"]
            yield instance

    def process(self, multi_stream: MultiStream) -> MultiStream:
        result = {}

        # optimization in to avoid double calculation of metrics
        # when aggregating results, if there is only one stream.
        if len(multi_stream) == 1:
            for stream_name, stream in multi_stream.items():
                result[stream_name] = Stream(
                    self.spread_results_one_stream, gen_kwargs={"stream": stream}
                )
            return MultiStream(result)

        mean_score = self.aggegate_results(multi_stream)
        result = {}
        for stream_name, stream in multi_stream.items():
            result[stream_name] = Stream(
                self.spread_results, gen_kwargs={"stream": stream, "score": mean_score}
            )

        return MultiStream(result)


class FromPredictionsAndOriginalData(StreamInitializerOperator):
    def zip(self, predictions, references):
        for prediction, original in zip(predictions, references):
            yield {**original, "prediction": prediction}

    def process(
        self, predictions: List[str], references: Iterable, split_name: str = "all"
    ) -> MultiStream:
        return MultiStream(
            {
                split_name: Stream(
                    self.zip,
                    gen_kwargs={"predictions": predictions, "references": references},
                )
            }
        )


# The task_data field in the schema is defined as
# Sequence({"key": Value(dtype="string"), "value": Value("string")})
# When receiving instances from this scheme, the keys and values are returned as two separate
# lists, and are converted to a dictionary.


class MetricRecipe(SequentialOperatorInitilizer):
    calc_confidence_intervals: bool = True

    def prepare(self):
        register_all_artifacts()
        self.steps = [
            FromPredictionsAndOriginalData(),
            Apply(
                "task_data",
                function="json.loads",
                to_field="task_data",
            ),
            ApplyOperatorsField(
                operators_field="postprocessors",
            ),
            SplitByValue(["group"]),
            ApplyMetric(
                "metrics",
                calc_confidence_intervals=self.calc_confidence_intervals,
            ),
            MultiStreamScoreMean(),
            MergeStreams(),
        ]


UNITXT_METRIC_SCHEMA = Features(
    {"predictions": Value("string"), "references": dict(UNITXT_DATASET_SCHEMA)}
)


def _compute(
    predictions: List[str],
    references: Iterable,
    flatten: bool = False,
    split_name: str = "all",
    calc_confidence_intervals: bool = True,
):
    _reset_env_local_catalogs()
    register_all_artifacts()
    recipe = MetricRecipe(calc_confidence_intervals=calc_confidence_intervals)

    multi_stream = recipe(
        predictions=predictions, references=references, split_name=split_name
    )

    if flatten:
        operator = FlattenInstances()
        multi_stream = operator(multi_stream)

    stream = multi_stream[split_name]
    return list(stream)


"""
The API of a metric service:
- MetricRequest: A single input request to the metrics service.
- MetricResponse: A response returned from a metrics service.
"""


class InstanceInput(Dataclass):
    """A single instance inputted to a metric service."""

    prediction: Any
    references: List[Any]
    additional_inputs: Optional[Dict] = None


class MetricRequest(Dataclass):
    """A request to a metrics service, includes a list of input instances."""

    instance_inputs: List[InstanceInput]


class MetricResponse(Dataclass):
    """A response produced by a metrics service, includes the computed scores."""

    # A list of instance score dictionaries. Each dictionary contains the
    # score names and score values for a single instance.
    instances_scores: List[Dict[str, Any]]
    # The global scores dictionary, containing global score names and values.
    # These are scores computed over the entire set of input instances, e.g.
    # an average over a score computed per instance.
    global_score: Dict[str, Any]


"""
Functionality for loading the remote metrics configuration from local environment variables.
"""

# A list of metrics to be executed remotely.
# For example: '["metrics.rag.context_relevance","metrics.rag.bert_k_precision"]'
# This value should be a valid json list
UNITXT_REMOTE_METRICS = "UNITXT_REMOTE_METRICS"

# The remote endpoint on which the remote metrics are available.
# For example, 'http://127.0.0.1:8000/compute'
UNITXT_REMOTE_METRICS_ENDPOINT = "UNITXT_REMOTE_METRICS_ENDPOINT"


def get_remote_metrics_names() -> List[str]:
    """Load the remote metrics names from an environment variable.

    Returns:
        List[str] - names of metrics to be executed remotely.
    """
    settings = get_settings()
    remote_metrics = settings.remote_metrics
    if remote_metrics:
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
    return remote_metrics


def get_remote_metrics_endpoint() -> str:
    """Load the remote metrics endpoint from an environment variable.

    Returns:
        str - The remote endpoint on which the remote metrics are available.
    """
    settings = get_settings()
    try:
        remote_metrics_endpoint = settings.remote_metrics_endpoint
    except AttributeError as e:
        raise RuntimeError(
            f"Unexpected None value for '{UNITXT_REMOTE_METRICS_ENDPOINT}'. "
            f"Running remote metrics requires defining an "
            f"endpoint in the environment variable '{UNITXT_REMOTE_METRICS_ENDPOINT}'."
        ) from e
    return remote_metrics_endpoint
