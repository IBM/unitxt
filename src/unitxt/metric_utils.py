import json
from copy import deepcopy
from typing import Any, Dict, Generator, Iterable, List, Optional

from datasets import Features, Value
from numpy import nanmean

from .dataclass import Dataclass
from .dict_utils import dict_set
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
    MergeStreams,
    RenameFields,
    SplitByNestedGroup,
)
from .register import _reset_env_local_catalogs, register_all_artifacts
from .schema import UNITXT_DATASET_SCHEMA
from .settings_utils import get_settings
from .stream import DynamicStream, MultiStream
from .struct_data_operators import LoadJson


class MultiStreamScoreMean(MultiStreamOperator):
    """Given a multi-stream where each stream is already scored globally, generate a nested global score for the whole multi-stream.

    The whole-ms-global-score is a nested structure, specifying (also) the individual global scores of the
    individual streams participating in the input multi_stream.
    The instances of all these individual streams are assumed to have the "group" field indicate the stream
    they belong to.
    Potentially, these individual streams were produced from a SplitByNestedGroup
    operator that did not use the full length of the value in field "group" of the instances, but only the
    first g components thereof, indicated by argument 'number_of_fusion_generations' of operator SplitByNestedGroup.
    At any rate, a distinguishing prefix of the "group" value is recorded, by operator SplitByNestedGroup, in the stream_name.
    The nested structure of the whole-ms-global-score is induced by these distinguishing prefixes,
    by virtue of the global score of each individual stream sitting in the nested whole-ms-global-score,
    deep in that dictionary, at the leaf lead to by a path being the distinguishing prefix indicated in the stream_name.
    Thus, the global score of the stream becomes a leaf (though a dict by itself) of the whole-ms-global-score.

    The ancestor nodes of the above leaves, in the whole-ms-global-score, contain each (in addition to dicts
    leading down to leaves) a field named "score" whose value is set to be the mean of the values
    sitting in field "score" of its immediate children nodes, and a field named "score_name" whose
    value is set to be "group_mean".

    When the input multistream consists of one single stream, it is returned as is, mainly for backward compatibility.
    """

    def update_intermediate_level_scores(self, level: dict) -> float:
        if "score" in level:
            return level["score"]
            # the global score of the stream participating in this MultiStream
        sub_scores = []
        for key in level:
            if isinstance(level[key], dict):
                sub_scores.append(self.update_intermediate_level_scores(level[key]))
        level.update({"score": nanmean(sub_scores), "score_name": "groups_mean"})
        return level["score"]

    def process(self, multi_stream: MultiStream) -> MultiStream:
        # each stream went through Metric which is a single-stream-operator , and ended up with all
        # its instance["score"]["global"] linking to the same single dict object.
        # Here we first generate a new, nested version, for the whole-ms-global_score, and then update
        # each stream's global score with the new version
        # but if only one stream in the multistream - we return it as is
        if len(multi_stream) == 1:
            return multi_stream
        global_score = {}
        first_instances = {}
        iterators = {}

        for stream_name, stream in multi_stream.items():
            iterators[stream_name] = iter(stream)
            try:
                first_instances[stream_name] = next(iterators[stream_name])
            except StopIteration:
                continue  # an empty stream, goto next stream
            instance = first_instances[stream_name]
            dict_set(
                dic=global_score,
                query=stream_name.split("~")[-1],
                value=deepcopy(instance["score"]["global"]),
                not_exist_ok=True,
            )

        self.update_intermediate_level_scores(global_score)
        # update the global_score object for each stream. Recall that all instances
        # in each stream link all to same python dict object
        for stream_name in multi_stream.keys():
            instance = first_instances[stream_name]
            instance["score"]["global"].clear()
            instance["score"]["global"].update(global_score)

        def never_peek_twice_generator(
            stream_name: str, first_instances: dict, iterators: dict
        ) -> Generator:
            while True:
                if stream_name in first_instances:
                    yield first_instances.pop(stream_name)
                try:
                    yield next(iterators[stream_name])
                except StopIteration:
                    return

        return MultiStream(
            {
                stream_name: DynamicStream(
                    never_peek_twice_generator,
                    gen_kwargs={
                        "stream_name": stream_name,
                        "first_instances": first_instances,
                        "iterators": iterators,
                    },
                )
                for stream_name in multi_stream.keys()
            }
        )


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


# The task_data field in the schema is defined as
# Sequence({"key": Value(dtype="string"), "value": Value("string")})
# When receiving instances from this scheme, the keys and values are returned as two separate
# lists, and are converted to a dictionary.

_post_process_steps = SequentialOperator(
    steps=[
        Copy(
            field="prediction",
            to_field="raw_prediction",
        ),
        Copy(
            field="references",
            to_field="raw_references",
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
        ),
    ]
)


class PostProcessRecipe(SequentialOperatorInitializer):
    def prepare(self):
        register_all_artifacts()
        self.steps = [
            FromPredictionsAndOriginalData(),
            _post_process_steps,
        ]


def _post_process(
    predictions: List[str],
    references: Iterable,
    split_name: str = "all",
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
    number_of_fusion_generations: int = 2

    def prepare(self):
        register_all_artifacts()
        self.steps = [
            FromPredictionsAndOriginalData(),
            LoadJson(field="task_data"),
            _post_process_steps,
            SplitByNestedGroup(
                field_name_of_group="group",
                number_of_fusion_generations=self.number_of_fusion_generations,
            ),
            ApplyMetric(
                "metrics",
                calc_confidence_intervals=self.calc_confidence_intervals,
            ),
            MultiStreamScoreMean(),
            MergeStreams(),
            RenameFields(
                field="raw_prediction",
                to_field="prediction",
            ),
            RenameFields(
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
