from typing import Dict, Iterable, List

from datasets import Features, Value

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


# The additional_inputs field in the schema is defined as
# Sequence({"key": Value(dtype="string"), "value": Value("string")})
# When receiving instances from this scheme, the keys and values are returned as two separate
# lists, and are converted to a dictionary.


def _from_key_value_pairs(key_value_list: Dict[str, list]) -> Dict[str, str]:
    return dict(zip(key_value_list["key"], key_value_list["value"]))


class MetricRecipe(SequentialOperatorInitilizer):
    calc_confidence_intervals: bool = True

    def prepare(self):
        register_all_artifacts()
        self.steps = [
            FromPredictionsAndOriginalData(),
            Apply(
                "additional_inputs",
                function=_from_key_value_pairs,
                to_field="additional_inputs",
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
