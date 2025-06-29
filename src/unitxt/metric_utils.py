import json
import re
import textwrap
from collections import defaultdict
from functools import lru_cache
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from datasets import Features, Value

from .dataclass import Dataclass
from .error_utils import Documentation, UnitxtError, error_context
from .operator import (
    InstanceOperator,
    MultiStreamOperator,
    SequentialOperator,
    SequentialOperatorInitializer,
    StreamInitializerOperator,
)
from .operators import (
    ApplyMetric,
    ApplyOperatorsField,
    ArtifactFetcherMixin,
    FlattenInstances,
    RecursiveCopy,
    Rename,
)
from .register import _reset_env_local_catalogs, register_all_artifacts
from .schema import UNITXT_DATASET_SCHEMA
from .settings_utils import get_constants, get_settings
from .stream import DynamicStream, MultiStream
from .struct_data_operators import LoadJson
from .text_utils import to_pretty_string
from .type_utils import isoftype
from .utils import recursive_copy

constants = get_constants()

DEFAULT_STREAM_NAME = "all_data"
DEFAULT_STREAM_SUBSET_SEPARATOR = ">>"


def nan_mean(scores):
    result = mean(score for score in scores if score == score)
    try:
        return float(result)
    except:
        return result


class FromPredictionsAndOriginalData(StreamInitializerOperator):
    def zip(self, predictions, references):
        for prediction, original in zip(predictions, references):
            if not isoftype(original, Dict[str, Any]):
                raise Exception(
                    f"The dataset passed for evaluation is not valid. Perhaps you passed a full dataset with multiple splits for evaluation instead of only the a single 'test' split. The offending instance: {original} "
                )

            yield {**original, "prediction": prediction}

    def process(
        self,
        predictions: List[str],
        references: Iterable,
        split_name: str = DEFAULT_STREAM_NAME,
    ) -> MultiStream:
        return MultiStream(
            {
                split_name: DynamicStream(
                    self.zip,
                    gen_kwargs={"predictions": predictions, "references": references},
                )
            }
        )


class DeleteTargetPrefix(InstanceOperator, ArtifactFetcherMixin):
    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if "metadata" in instance["task_data"]:
            target_prefix = self.get_artifact(
                instance["task_data"]["metadata"]["template"]
            ).target_prefix
            if target_prefix is not None and len(target_prefix) > 0:
                target_prefix = target_prefix.format(**instance["task_data"])
                pattern = rf"^\s*{re.escape(target_prefix)}\s*"
                instance["prediction"] = re.sub(pattern, "", instance["prediction"])
        return instance


_post_process_steps = SequentialOperator(
    steps=[
        RecursiveCopy(
            field="prediction",
            to_field="raw_prediction",
        ),
        RecursiveCopy(
            field="references",
            to_field="raw_references",
            dont_apply_to_streams=[constants.inference_stream],
        ),
        RecursiveCopy(
            field="source",
            to_field="task_data/source",
        ),
        DeleteTargetPrefix(),
        ApplyOperatorsField(
            operators_field="postprocessors",
        ),
        RecursiveCopy(
            field="prediction",
            to_field="processed_prediction",
        ),
        RecursiveCopy(
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
                    + DEFAULT_STREAM_SUBSET_SEPARATOR
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
    origin, subset_group = stream_name.split(DEFAULT_STREAM_SUBSET_SEPARATOR)
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
                all_num_of_instances = []
                for k, v in dic.items():
                    score = recursive_mean(v)
                    if score is not None:
                        all_scores.append(score["score"])
                        if "num_of_instances" in score:
                            all_num_of_instances.append(score["num_of_instances"])
                        result[k] = score

                result["score"] = nan_mean(all_scores)
                result["score_name"] = "subsets_mean"
                if all_num_of_instances:
                    result["num_of_instances"] = sum(all_num_of_instances)

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
                    "subsets_mean": score["subsets"]["score"],
                }
                if "num_of_instances" in score["subsets"]:
                    score["global"]["num_of_instances"] = score["subsets"][
                        "num_of_instances"
                    ]

            sorted_instances = []
            for key in sorted(stream_instances.keys()):
                instance = stream_instances[key]
                instance["score"].update(recursive_copy(score))
                sorted_instances.append(instance)
            result[stream_name] = sorted_instances

        return MultiStream.from_iterables(result, copying=True)


class PostProcessRecipe(SequentialOperatorInitializer):
    def prepare(self):
        register_all_artifacts()
        self.steps = [
            FromPredictionsAndOriginalData(),
            LoadJson(field="task_data"),
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
            RecursiveCopy(
                field="source",
                to_field="task_data/source",
            ),
        ]


UNITXT_METRIC_SCHEMA = Features(
    {"predictions": Value("string"), "references": dict(UNITXT_DATASET_SCHEMA)}
)


class GlobalScores(dict):
    """GlobalScores is a dictionary-based class designed to handle and transform metric results into a structured format.

    Args:
        score (float):
            The main score value.
        score_name (str):
            The name of the main score.
    """

    @property
    def score(self):
        return self["score"]

    @property
    def score_name(self):
        return self["score_name"]

    def to_df(self):
        """Transforms a dictionary of results into a pandas dataframe.

        Transforms a dictionary of results into a dataframe with score_name as the index,
        and columns for score, ci_low, and ci_high. Handles cases where confidence intervals are missing.

        Returns:
            pd.DataFrame: A dataframe with the extracted information, indexed by score_name.
        """
        import pandas as pd

        rows = []

        # Extract data based on score names
        for key, value in self.items():
            if key.endswith("_ci_low") or key.endswith("_ci_high"):
                continue  # Skip confidence interval keys for now

            if isinstance(value, (int, float)):  # Only consider numerical scores
                score_name = key
                ci_low = self.get(f"{key}_ci_low", None)
                ci_high = self.get(f"{key}_ci_high", None)

                rows.append(
                    {
                        "score_name": score_name,
                        "score": value,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

        df = pd.DataFrame(rows)
        return df.set_index("score_name")

    def __repr__(self):
        return to_pretty_string(self, float_format=".2g")

    @property
    def summary(self):
        df = self.to_df().round(2).fillna("")
        df = df.sort_index()
        df = df.drop("num_of_instances", axis=0)
        df = df.reset_index()
        score_name = self["score_name"]
        num_of_instances = self["num_of_instances"]
        return (
            df.to_markdown(index=False)
            + f"\nMain Score: {score_name}\nNum Instances: {num_of_instances}"
        )


class SubsetsScores(dict):
    def __repr__(self):
        return to_pretty_string(self, float_format=".2g")

    @property
    def summary(self):
        rows = []
        data = self
        rows = []
        all_group_types = set()

        def walk_subsets(node, subset_path):
            # Check if this node represents a subset level by checking "score" and "score_name"
            is_subset_node = "score" in node and "score_name" in node

            # Extract subset-level info if this is a subset node
            if is_subset_node:
                subset_score = node.get("score", "")
                subset_score_name = node.get("score_name", "")
                subset_ci_low = node.get("score_ci_low", "")
                subset_ci_high = node.get("score_ci_high", "")
                subset_num_instances = node.get("num_of_instances", "")

                # Check for groups at this level
                groups = node.get("groups", {})

                if groups:
                    # If there are groups, we create one row per group entry
                    for group_type, group_dict in groups.items():
                        for group_name, group_metrics in group_dict.items():
                            g_score = group_metrics.get("score", subset_score)
                            g_score_name = group_metrics.get(
                                "score_name", subset_score_name
                            )
                            g_ci_low = group_metrics.get("score_ci_low", subset_ci_low)
                            g_ci_high = group_metrics.get(
                                "score_ci_high", subset_ci_high
                            )
                            g_num_instances = group_metrics.get(
                                "num_of_instances", subset_num_instances
                            )

                            all_group_types.add(group_type)

                            row = {
                                "subset": ".".join(subset_path)
                                if subset_path
                                else "ALL",
                                "score": g_score,
                                "score_name": g_score_name,
                                "score_ci_low": g_ci_low,
                                "score_ci_high": g_ci_high,
                                "num_of_instances": g_num_instances,
                                group_type: str(group_name),
                            }
                            rows.append(row)
                else:
                    # No groups, just one row for this subset node
                    row = {
                        "subset": ".".join(subset_path) if subset_path else "ALL",
                        "score": subset_score,
                        "score_name": subset_score_name,
                        "score_ci_low": subset_ci_low,
                        "score_ci_high": subset_ci_high,
                        "num_of_instances": subset_num_instances,
                    }
                    rows.append(row)

            # Now check for deeper subsets: any key in node that leads to another dict with "score" and "score_name"
            # or even if it doesn't have score, we still recurse to find deeper subsets.
            for k, v in node.items():
                if isinstance(v, dict) and k != "groups":
                    # If v is a dict, recurse
                    # We'll attempt to go deeper since subsets can be arbitrary depth
                    # We do not require v to have score/score_name at this time, recursion can find deeper ones.
                    walk_subsets(v, [*subset_path, k])

        # Start recursion from top-level
        walk_subsets(data, [])

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Ensure columns exist for all group types
        for gt in all_group_types:
            if gt not in df.columns:
                df[gt] = ""

        # Replace NaN with ""
        df = df.fillna("")

        # Remove columns that are all empty strings
        df = df.drop(columns=[col for col in df.columns if df[col].eq("").all()])

        # Attempt to order columns in a logical manner:
        # subset first, then any group type columns, then score fields
        fixed_cols = [
            "subset",
            "score",
            "score_name",
            "score_ci_low",
            "score_ci_high",
            "num_of_instances",
        ]
        group_type_cols = [
            c for c in df.columns if c not in fixed_cols and c != "subset"
        ]
        order = [
            "subset",
            *group_type_cols,
            "score",
            "score_name",
            "score_ci_low",
            "score_ci_high",
            "num_of_instances",
        ]
        order = [c for c in order if c in df.columns]
        df = df[order]

        return df.to_markdown(index=False)


class GroupsScores(dict):
    """A dictionary subclass to store and manage group scores.

    This class provides a property to summarize the scores and a custom
    string representation for pretty-printing.

    """

    @property
    def summary(self):
        """A property to get a summary of the group scores."""
        data = self
        # Desired metric columns
        metric_cols = [
            "score",
            "score_name",
            "score_ci_low",
            "score_ci_high",
            "num_of_instances",
        ]
        output_lines = []

        for scenario_key, scenario_data in data.items():
            # scenario_key could be a single string or a tuple of strings
            if isinstance(scenario_key, tuple):
                scenario_groups = scenario_key
            else:
                scenario_groups = (scenario_key,)

            # Build rows for this scenario
            rows = []
            for group_name_key, metrics in scenario_data.items():
                # group_name_key should match the structure of scenario_groups
                if isinstance(group_name_key, tuple):
                    group_names = group_name_key
                else:
                    group_names = (group_name_key,)

                # Create a row with group columns and metric columns
                row = {}
                for g_type, g_name in zip(scenario_groups, group_names):
                    row[g_type] = str(g_name)

                # Add desired metrics
                for mcol in metric_cols:
                    row[mcol] = metrics.get(mcol, "")

                rows.append(row)

            # Convert this scenario's rows to a DataFrame
            if rows:
                df = pd.DataFrame(rows)
            else:
                # No rows means empty DataFrame
                df = pd.DataFrame(columns=list(scenario_groups) + metric_cols)

            # Fill NaN with ""
            df = df.fillna("")

            # Remove columns that are entirely empty
            df = df.drop(columns=[col for col in df.columns if df[col].eq("").all()])

            # Order columns: group types first (in the order they appear in scenario_groups), then metrics
            final_cols = [col for col in scenario_groups if col in df.columns] + [
                col for col in metric_cols if col in df.columns
            ]
            df = df[final_cols]

            # Title for this scenario
            if len(scenario_groups) == 1:
                title = f"# Group By: {scenario_groups[0]}"
            else:
                title = "# Group By: " + ", ".join(scenario_groups)
            output_lines.append(title)

            if not df.empty:
                output_lines.append(df.to_markdown(index=False))
            else:
                output_lines.append("_No matching rows_")

            output_lines.append("")

        return "\n".join(output_lines)

    def __repr__(self):
        return to_pretty_string(self, float_format=".2g")


class InstanceScores(list):
    def __init__(self, instances):
        self.original_instances = instances
        instance_scores = []
        for instance in instances:
            instance = instance.copy()
            scores = instance.pop("score")
            task_data = instance.pop("task_data")
            instance_scores.append(
                {
                    **task_data,
                    **instance,
                    **scores["instance"],
                }
            )
        super().__init__(instance_scores)

    def to_df(self, flatten=True, columns=None):
        """Transforms the stored results into a pandas DataFrame.

        Args:
            flatten (bool, optional): Determines whether to use the flattened list of results (`self`)
                or the original instances (`self.original_instances`). Defaults to True.
            columns (list, optional): A list of column names to select from the resulting DataFrame.
                If None, all columns are included. Defaults to None.

        Returns:
            pandas.DataFrame: A DataFrame containing the transformed results. If `columns` is specified,
            only the specified columns are included.

        Raises:
            KeyError: If any specified column in `columns` does not exist in the DataFrame.
        """
        from pandas import DataFrame

        if flatten:
            df = DataFrame(self)
        else:
            df = DataFrame(self.original_instances)
        if columns is not None:
            return df[columns]
        return df

    def _to_markdown(self, df, max_col_width=30, **kwargs):
        def wrap_column(series, max_width=30):
            """Wraps string values in a Pandas Series to a maximum width."""
            return series.apply(
                lambda x: "\n".join(
                    textwrap.fill(line, width=max_width) for line in str(x).splitlines()
                )
            )

        wrapped_df = df.copy()
        for col in wrapped_df.columns:
            wrapped_df[col] = wrap_column(wrapped_df[col], max_col_width)
        return wrapped_df.to_markdown(**kwargs)

    def to_markdown(self, flatten=True, columns=None, max_col_width=30, **kwargs):
        return self._to_markdown(self.to_df(flatten, columns), max_col_width, **kwargs)

    @property
    def summary(self):
        df = self.to_df(
            flatten=False,
            columns=[
                "source",
                "prediction",
                "processed_prediction",
                "references",
                "processed_references",
                "score",
            ],
        ).head()
        df["score_name"] = df["score"].apply(lambda x: x["instance"]["score_name"])
        df["all_scores"] = df["score"].apply(
            lambda x: "\n".join(f"{k}: {v}" for k, v in x["instance"].items())
        )
        df["score"] = df["score"].apply(lambda x: x["instance"]["score"])

        return self._to_markdown(df)

    def __repr__(self):
        return to_pretty_string(self, float_format=".2g")


class EvaluationResults(list):
    def __init__(self, *args, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata if metadata is not None else {}

    @property
    def global_scores(self):
        return GlobalScores(self[0]["score"]["global"])

    @property
    def instance_scores(self) -> InstanceScores:
        return InstanceScores(self)

    @property
    def groups_scores(self):
        if "groups" not in self[0]["score"]:
            raise UnitxtError(
                "Groups scores not found try using group_by in the recipe",
                additional_info_id=Documentation.EVALUATION,
            )
        return GroupsScores(self[0]["score"]["groups"])

    @property
    def subsets_scores(self):
        if "subsets" not in self[0]["score"]:
            raise UnitxtError(
                "Subsets scores not found try using Benchmark",
                additional_info_id=Documentation.BENCHMARKS,
            )
        return SubsetsScores(self[0]["score"]["subsets"])


def _compute(
    predictions: List[Any],
    references: Iterable,
    flatten: bool = False,
    split_name: str = DEFAULT_STREAM_NAME,
    calc_confidence_intervals: bool = True,
):
    _reset_env_local_catalogs()
    register_all_artifacts()
    recipe = MetricRecipe(calc_confidence_intervals=calc_confidence_intervals)

    with error_context(stage="Metric Processing"):
        multi_stream = recipe(
            predictions=predictions, references=references, split_name=split_name
        )

        if flatten:
            operator = FlattenInstances()
            multi_stream = operator(multi_stream)

        stream = multi_stream[split_name]
    return EvaluationResults(stream)


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
