import json
from collections import defaultdict
from functools import lru_cache
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from datasets import Features, Value

from .dataclass import Dataclass
from .operator import (
    MultiStreamOperator,
    SequentialOperator,
    SequentialOperatorInitializer,
    StreamInitializerOperator,
)
from .operators import (
    ApplyMetric,
    ApplyOperatorsField,
    Copy,
    FlattenInstances,
    Rename,
)
from .register import _reset_env_local_catalogs, register_all_artifacts
from .schema import UNITXT_DATASET_SCHEMA
from .settings_utils import get_constants, get_settings
from .stream import DynamicStream, MultiStream
from .struct_data_operators import LoadJson
from .utils import deepcopy

constants = get_constants()


def nan_mean(scores):
    return mean(score for score in scores if score == score)


class FromPredictionsAndOriginalData(StreamInitializerOperator):
    def zip(self, predictions, references):
        for prediction, original in zip(predictions, references):
            yield {**original, "prediction": prediction}

    def process(
        self, predictions: List[str], references: Iterable, split_name: str = "all"
    ) -> MultiStream:
        return MultiStream(
            {
                split_name: DynamicStream(
                    self.zip,
                    gen_kwargs={"predictions": predictions, "references": references},
                )
            }
        )


_post_process_steps = SequentialOperator(
    steps=[
        Copy(
            field="prediction",
            to_field="raw_prediction",
        ),
        Copy(
            field="references",
            to_field="raw_references",
            dont_apply_to_streams=[constants.inference_stream],
        ),
        Copy(
            field="source",
            to_field="task_data/source",
        ),
        ApplyOperatorsField(
            operators_field="postprocessors",
        ),
        Copy(
            field="prediction",
            to_field="processed_prediction",
        ),
        Copy(
            field="references",
            to_field="processed_references",
            dont_apply_to_streams=[constants.inference_stream],
        ),
    ]
)


@lru_cache(maxsize=None)
def group_str(json_str):
    data = json.loads(json_str)
    return ",".join(f"{k}:{v}" for k, v in data.items())


class SplitSubsetsAndGroups(MultiStreamOperator):
    """Splits a MultiStream that is small - for metrics, hence: whole stream can sit in memory, split by the value of field 'group'.

    Args:
        number_of_fusion_generations: int

    the value in field group is of the form "sourcen/sourcenminus1/..." describing the sources in which the instance sat
    when these were fused, potentially several phases of fusion. the name of the most recent source sits first in this value.
    (See BaseFusion and its extensions)
    subsets_depth  specifies the depth of the prefix by which to split the stream.
    """

    subsets_field: str = "subset"
    groups_field: str = "groups"
    subset_depth: Optional[int] = None

    def process(self, multi_stream: MultiStream) -> MultiStream:
        result = defaultdict(list)

        for stream_name, stream in multi_stream.items():
            for i, instance in enumerate(stream):
                instance["__idx__"] = i

                for field in [self.subsets_field, self.groups_field]:
                    if field not in instance:
                        raise ValueError(
                            f"Field {field} is missing from instance {instance}"
                        )

                subset_stream_name = (
                    stream_name
                    + "://"
                    + "/".join(instance[self.subsets_field][: self.subset_depth])
                )

                result[subset_stream_name].append(instance)

                for group in instance[self.groups_field]:
                    result[subset_stream_name + "?" + group_str(group)].append(instance)

        return MultiStream.from_iterables(result, copying=True)


@lru_cache(maxsize=None)
def group_str_to_key_value(group_str):
    keys = []
    values = []
    for k_v in group_str.split(","):
        k, v = k_v.split(":")
        if v.isdigit():
            v = int(v)
        keys.append(k)
        values.append(v)

    if len(keys) == 1:
        key = keys[0]
    else:
        key = tuple(keys)

    if len(values) == 1:
        value = values[0]
    else:
        value = tuple(values)

    return key, value


@lru_cache(maxsize=None)
def stream_name_to_origin_subset_group(stream_name):
    origin, subset_group = stream_name.split("://")
    if "?" in subset_group:
        subset, group = subset_group.split("?")
    else:
        subset, group = subset_group, None
    return origin, subset, group


class JoinSubsetsAndGroups(MultiStreamOperator):
    def process(self, multi_stream: MultiStream) -> MultiStream:
        instances = defaultdict(dict)
        global_scores = defaultdict(dict)

        for stream_name, stream in multi_stream.items():
            origin, subset, group = stream_name_to_origin_subset_group(stream_name)

            for i, instance in enumerate(stream):
                global_score = instance["score"].pop("global")

                idx = instance.pop("__idx__")
                if idx not in instances[origin]:
                    instances[origin][idx] = instance

                # from here below setting the global scores from that stream
                # can be done with first instance only
                if i > 0:
                    continue

                if not group and not subset:
                    global_scores[origin]["global"] = global_score
                else:
                    path = []

                    if subset:
                        path += ["subsets", *subset.split("/")]

                    if group:
                        key, value = group_str_to_key_value(group)
                        path += ["groups", key, value]

                    target = global_scores[origin]
                    for part in path[:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    target[path[-1]] = global_score

        # the leafs always have score_name and score
        def recursive_mean(dic):
            if isinstance(dic, dict):
                if "score" in dic and "score_name" in dic:
                    return dic

                result = {}
                all_scores = []
                for k, v in dic.items():
                    score = recursive_mean(v)
                    if score is not None:
                        all_scores.append(score["score"])
                        result[k] = score

                result["score"] = nan_mean(all_scores)
                result["score_name"] = "subsets_mean"

                if result:
                    return result

            return None

        result = {}
        for stream_name, stream_instances in instances.items():
            score = global_scores[stream_name]

            if "subsets" in score:
                score["subsets"] = recursive_mean(score["subsets"])
                score["global"] = {
                    "score": score["subsets"]["score"],
                    "score_name": score["subsets"]["score_name"],
                }

            sorted_instances = []
            for key in sorted(stream_instances.keys()):
                instance = stream_instances[key]
                instance["score"].update(deepcopy(score))
                sorted_instances.append(instance)
            result[stream_name] = sorted_instances

        return MultiStream.from_iterables(result, copying=True)


class PostProcessRecipe(SequentialOperatorInitializer):
    def prepare(self):
        register_all_artifacts()
        self.steps = [
            FromPredictionsAndOriginalData(),
            _post_process_steps,
        ]


def _inference_post_process(
    predictions: List[str],
    references: Iterable,
    split_name: str = constants.inference_stream,
):
    _reset_env_local_catalogs()
    register_all_artifacts()
    recipe = PostProcessRecipe()

    multi_stream = recipe(
        predictions=predictions, references=references, split_name=split_name
    )

    return [instance["processed_prediction"] for instance in multi_stream[split_name]]


class MetricRecipe(SequentialOperatorInitializer):
    calc_confidence_intervals: bool = True
    subset_depth: int = 2

    def prepare(self):
        register_all_artifacts()
        self.steps = [
            FromPredictionsAndOriginalData(),
            LoadJson(field="task_data"),
            _post_process_steps,
            SplitSubsetsAndGroups(
                subset_depth=self.subset_depth,
            ),
            ApplyMetric(
                "metrics",
                calc_confidence_intervals=self.calc_confidence_intervals,
            ),
            JoinSubsetsAndGroups(),
            Rename(
                field="raw_prediction",
                to_field="prediction",
            ),
            Rename(
                field="raw_references",
                to_field="references",
            ),
            Copy(
                field="source",
                to_field="task_data/source",
            ),
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
