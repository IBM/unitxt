from dataclasses import field
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import datasets
import evaluate
from datasets import Features, Sequence, Value

from .artifact import __file__ as _
from .blocks import __file__ as _
from .card import __file__ as _
from .catalog import __file__ as _
from .collections import __file__ as _
from .dataclass import __file__ as _
from .dict_utils import __file__ as _
from .file_utils import __file__ as _
from .formats import __file__ as _
from .fusion import __file__ as _
from .generator_utils import __file__ as _
from .hf_utils import __file__ as _
from .instructions import __file__ as _
from .load import __file__ as _
from .loaders import __file__ as _
from .metrics import __file__ as _
from .normalizers import __file__ as _
from .operator import (
    MultiStreamOperator,
    SequentialOperator,
    SequentialOperatorInitilizer,
    StreamInitializerOperator,
)
from .operator import __file__ as _
from .operators import (
    Apply,
    ApplyOperatorsField,
    ApplyStreamOperatorsField,
    FlattenInstances,
    MergeStreams,
    SplitByValue,
)
from .operators import __file__ as _
from .processors import __file__ as _
from .random_utils import __file__ as _
from .recipe import __file__ as _
from .register import __file__ as _
from .register import _reset_env_local_catalogs, register_all_artifacts
from .renderers import __file__ as _
from .schema import __file__ as _
from .split_utils import __file__ as _
from .splitters import __file__ as _
from .standard import __file__ as _
from .stream import MultiStream, Stream
from .stream import __file__ as _
from .task import __file__ as _
from .templates import __file__ as _
from .text_utils import __file__ as _
from .type_utils import __file__ as _
from .utils import __file__ as _
from .validate import __file__ as _
from .version import __file__ as _


class MultiStreamScoreMean(MultiStreamOperator):
    def aggegate_results(self, multi_stream: MultiStream):
        scores = []
        for stream in multi_stream.values():
            instance = stream.peak()
            scores.append(instance["score"]["global"]["score"])

        from statistics import mean

        return mean(scores)

    def spread_results(self, stream: Stream, score: float):
        for instance in stream:
            instance["score"]["global"]["groups_mean_score"] = score
            yield instance

    def process(self, multi_stream: MultiStream) -> MultiStream:
        mean_score = self.aggegate_results(multi_stream)

        result = {}
        for stream_name, stream in multi_stream.items():
            result[stream_name] = Stream(self.spread_results, gen_kwargs={"stream": stream, "score": mean_score})

        return MultiStream(result)


class FromPredictionsAndOriginalData(StreamInitializerOperator):
    def zip(self, predictions, references):
        for prediction, original in zip(predictions, references):
            yield {**original, "prediction": prediction}

    def process(self, predictions: List[str], references: Iterable, split_name: str = "all") -> MultiStream:
        return MultiStream(
            {split_name: Stream(self.zip, gen_kwargs={"predictions": predictions, "references": references})}
        )


from .schema import UNITXT_DATASET_SCHEMA


def _from_key_value_pairs(key_value_list: Dict[str, list]) -> Dict[str, str]:
    return dict([(key, value) for key, value in zip(key_value_list["key"], key_value_list["value"])])


class MetricRecipe(SequentialOperatorInitilizer):
    def prepare(self):
        register_all_artifacts()
        self.steps = [
            FromPredictionsAndOriginalData(),
            Apply("additional_inputs", function=_from_key_value_pairs, to_field="additional_inputs"),
            ApplyOperatorsField(
                inputs_fields=["prediction", "references"],
                fields_to_treat_as_list=["references"],
                operators_field="postprocessors",
                default_operators=["processors.to_string_stripped"],
            ),
            SplitByValue(["group"]),
            ApplyStreamOperatorsField(
                "metrics",
                reversed=True,
            ),
            MultiStreamScoreMean(),
            MergeStreams(),
        ]


UNITXT_METRIC_SCHEMA = Features({"predictions": Value("string"), "references": dict(UNITXT_DATASET_SCHEMA)})


def _compute(predictions: List[str], references: Iterable, flatten: bool = False, split_name: str = "all"):
    _reset_env_local_catalogs()
    register_all_artifacts()
    recipe = MetricRecipe()
    print("References", references)
    multi_stream = recipe(predictions=predictions, references=references, split_name=split_name)

    if flatten:
        operator = FlattenInstances()
        multi_stream = operator(multi_stream)

    stream = multi_stream[split_name]
    return list(stream)


# TODO: currently we have two classes with this name. metric.Metric and matrics.Metric...
# @evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Metric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="_DESCRIPTION",
            citation="_CITATION",
            # inputs_description=_KWARGS_DESCRIPTION,
            features=UNITXT_METRIC_SCHEMA,
            codebase_urls=["https://"],
            reference_urls=[
                "https://",
                "https://",
            ],
        )

    def _compute(self, predictions: List[str], references: Iterable, flatten: bool = False, split_name: str = "all"):
        try:
            from unitxt.dataset import (
                get_dataset_artifact as get_dataset_artifact_installed,
            )

            unitxt_installed = True
        except ImportError:
            unitxt_installed = False

        if unitxt_installed:
            from unitxt.metric import _compute as _compute_installed

            return _compute_installed(
                predictions=predictions, references=references, flatten=flatten, split_name=split_name
            )
        else:
            return _compute(predictions=predictions, references=references, flatten=flatten, split_name=split_name)
