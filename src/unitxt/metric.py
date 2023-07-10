#####
# imports for hf system:
#####
from .artifact import __file__ as _
from .blocks import __file__ as _
from .card import __file__ as _
from .catalog import __file__ as _
from .collections import __file__ as _
from .common import __file__ as _
from .file_utils import __file__ as _

# from .fusion import __file__
from .generator_utils import __file__ as _
from .instructions import __file__ as _
from .loaders import __file__ as _
from .load import __file__ as _
from .metrics import __file__ as _
from .normalizers import __file__ as _
from .operator import __file__ as _
from .operators import __file__ as _
from .processors import __file__ as _
from .recipe import __file__ as _
from .register import __file__ as _
from .splitters import __file__ as _
from .split_utils import __file__ as _
from .stream import __file__ as _
from .task import __file__ as _
from .templates import __file__ as _
from .text_utils import __file__ as _
from .schema import __file__ as _

# from .utilize import __file__ as _
# from .validate import __file__ as _
#############

from .stream import MultiStream, Stream

from .operator import SequntialOperator, SequntialOperatorInitilizer, MultiStreamOperator, StreamInitializerOperator

from .operators import (
    ApplyValueOperatorsField,
    ApplyStreamOperatorsField,
    SplitByValue,
    MergeStreams,
    FlattenInstances,
)

import evaluate
import datasets

from datasets import (
    Features,
    Value,
    Sequence,
)

from dataclasses import field
from typing import List, Union, Dict, Optional, Generator, Any, Iterable


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


class MetricRecipe(SequntialOperatorInitilizer):
    def prepare(self):
        self.steps = [
            FromPredictionsAndOriginalData(),
            ApplyValueOperatorsField(
                value_field="prediction", operators_field="processors", default_operators=["to_string"]
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


# @evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class UnitextMetric(evaluate.Metric):
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
        recipe = MetricRecipe()

        multi_stream = recipe(predictions=predictions, references=references, split_name=split_name)

        if flatten:
            operator = FlattenInstances()
            multi_stream = operator(multi_stream)

        stream = multi_stream[split_name]

        return list(stream)
