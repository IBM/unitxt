import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import evaluate
import nltk
import numpy

from .operator import (
    MultiStreamOperator,
    SequntialOperator,
    SingleStreamOperator,
    StreamingOperator,
    StreamInstanceOperator,
)
from .operators import CopyFields
from .stream import MultiStream, Stream

nltk.download("punkt")


def absrtact_factory():
    return {}


def abstract_field():
    return field(default_factory=absrtact_factory)


class UpdateStream(StreamInstanceOperator):
    update: dict

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        instance.update(self.update)
        return instance


# TODO: currently we have two classes with this name. metric.Metric and matrics.Metric...
class Metric(ABC):
    @property
    @abstractmethod
    def main_score(self):
        pass


class GlobalMetric(SingleStreamOperator, Metric):
    def process(self, stream: Stream, stream_name: str = None) -> Generator:
        references = []
        predictions = []
        global_score = {}

        instances = []

        for instance in stream:
            if "score" not in instance:
                instance["score"] = {"global": global_score, "instance": {}}
            else:
                global_score = instance["score"]["global"]

            refs, pred = instance["references"], instance["prediction"]

            instance_score = self._compute([refs], [pred])
            instance["score"]["instance"].update(instance_score)

            references.append(refs)
            predictions.append(pred)
            instances.append(instance)

        result = self._compute(references, predictions)

        global_score.update(result)

        for instance in instances:
            instance["score"]["global"] = global_score
            yield instance

    def _compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        result = self.compute(references, predictions)
        result["score"] = result[self.main_score]
        return result

    @abstractmethod
    def compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        pass


class InstanceMetric(SingleStreamOperator, Metric):
    implemented_reductions: List[str] = field(default_factory=lambda: ["mean"])

    @property
    @abstractmethod
    def reduction_map(self) -> dict:
        pass

    def process(self, stream: Stream, stream_name: str = None) -> Generator:
        global_score = {}
        instances = []

        for instance in stream:
            refs, pred = instance["references"], instance["prediction"]

            instance_score = self._compute(refs, pred)

            if "score" not in instance:
                instance["score"] = {"global": global_score, "instance": {}}
            else:
                global_score = instance["score"]["global"]

            instance["score"]["instance"].update(instance_score)

            instances.append(instance)

        for reduction, fields in self.reduction_map.items():
            assert (
                reduction in self.implemented_reductions
            ), f"Reduction {reduction} is not implemented, use one of {self.implemented_reductions}"

            if reduction == "mean":
                from statistics import mean

                for field in fields:
                    global_score[field] = mean([instance["score"]["instance"][field] for instance in instances])
                    if field == self.main_score:
                        global_score["score"] = global_score[field]

        for instance in instances:
            yield instance

    def _compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        result = self.compute(references=references, predictions=predictions)
        result["score"] = result[self.main_score]
        return result

    @abstractmethod
    def compute(self, references: List[str], prediction: str) -> dict:
        pass


class Squad(GlobalMetric):
    _metric = None
    reduction_map = {"mean": ["f1"]}
    main_score = "f1"
    metric = "squad"

    def prepare(self):
        super(Squad, self).prepare()
        self._metric = evaluate.load(self.metric)

    def compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        ids = [str(uuid.uuid4()).replace("-", "") for _ in range(len(predictions))]
        formatted_predictions = [
            {"prediction_text": prediction, "id": ids[i]} for i, prediction in enumerate(predictions)
        ]
        formatted_references = [
            {"answers": {"answer_start": [-1], "text": reference}, "id": ids[i]}
            for i, reference in enumerate(references)
        ]

        return self._metric.compute(predictions=formatted_predictions, references=formatted_references)


class SingleReferenceInstanceMetric(InstanceMetric):
    def _compute(self, references: List[str], prediction: str) -> dict:
        result = self.compute(references[0], prediction)
        result["score"] = result[self.main_score]
        return result

    @abstractmethod
    def compute(self, reference, prediction: str) -> dict:
        pass


class Accuracy(SingleReferenceInstanceMetric):
    reduction_map = {"mean": ["accuracy"]}
    main_score = "accuracy"

    def compute(self, reference, prediction: str) -> dict:
        return {"accuracy": float(str(reference) == str(prediction))}


class MetricPipeline(MultiStreamOperator, Metric):
    main_score: str = None
    preprocess_steps: Optional[List[StreamingOperator]] = field(default_factory=list)
    postpreprocess_steps: Optional[List[StreamingOperator]] = field(default_factory=list)
    metric: Metric = None

    def verify(self):
        assert self.main_score is not None, "main_score is not set"

    def prepare(self):
        super().prepare()
        self.prepare_score = CopyFields(
            field_to_field=[
                [f"score/instance/{self.main_score}", "score/instance/score"],
                [f"score/global/{self.main_score}", "score/global/score"],
            ],
            use_query=True,
        )

    def process(self, multi_stream: MultiStream) -> MultiStream:
        for step in self.preprocess_steps:
            multi_stream = step(multi_stream)
        multi_stream = self.metric(multi_stream)
        for step in self.postpreprocess_steps:
            multi_stream = step(multi_stream)
        multi_stream = self.prepare_score(multi_stream)
        return multi_stream


class HuggingfaceMetric(GlobalMetric):
    metric_name: str = None
    main_score: str = None
    scale: float = 1.0

    def prepare(self):
        super().prepare()
        self.metric = evaluate.load(self.metric_name)

    def compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        result = self.metric.compute(predictions=predictions, references=references)
        if self.scale != 1.0:
            for key in result:
                if isinstance(result[key], float):
                    result[key] /= self.scale
        return result


class F1(GlobalMetric):
    _metric = None
    main_score = "f1_macro"
    average = None  # Report per class then aggregate by mean
    metric = "f1"

    def prepare(self):
        super(F1, self).prepare()
        self._metric = evaluate.load(self.metric)

    def get_str_id(self, str):
        if str not in self.str_to_id:
            id = len(self.str_to_id)
            self.str_to_id[str] = id
            self.id_to_str[id] = str
        return self.str_to_id[str]

    def compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        assert all(
            len(reference) == 1 for reference in references
        ), "One single reference per predictition are allowed in F1 metric"
        self.str_to_id = {}
        self.id_to_str = {}
        formatted_references = [self.get_str_id(reference[0]) for reference in references]
        unique_labels = self.str_to_id.keys()
        formatted_predictions = [self.get_str_id(prediction) for prediction in predictions]
        labels = list(set(formatted_references))
        result = self._metric.compute(
            predictions=formatted_predictions, references=formatted_references, labels=labels, average=self.average
        )
        if isinstance(result["f1"], numpy.ndarray):
            from statistics import mean

            final_result = {self.main_score: mean(result["f1"])}
            for i, label in enumerate(labels):
                final_result["f1_" + self.id_to_str[label]] = result["f1"][i]
        else:
            final_result = {self.main_score: result["f1"]}
        return final_result


class F1Micro(F1):
    main_score = "f1_micro"
    average = "micro"


class F1Macro(F1):
    main_score = "f1_macro"


class F1MultiLabel(GlobalMetric):
    _metric = None
    main_score = "f1_macro"
    average = None  # Report per class then aggregate by mean
    seperator = ","

    def prepare(self):
        super(F1MultiLabel, self).prepare()
        self._metric = evaluate.load("f1", "multilabel")

    def add_str_to_id(self, str):
        if not str in self.str_to_id:
            id = len(self.str_to_id)
            self.str_to_id[str] = id
            self.id_to_str[id] = str
        return

    def get_one_hot_vector(self, labels: List[str]):
        result = [0] * len(self.str_to_id)
        for label in labels:
            if label in self.str_to_id:
                result[self.str_to_id[label]] = 1
        return result

    def compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        self.str_to_id = {}
        self.id_to_str = {}
        labels = list(set([label for reference in references for label in reference]))
        for label in labels:
            assert (
                not self.seperator in label
            ), "Reference label (f{label}) can not contain multi label seperator (f{self.seperator}) "
            self.add_str_to_id(label)
        formatted_references = [self.get_one_hot_vector(reference) for reference in references]
        split_predictions = [
            [label.strip() for label in prediction.split(self.seperator)] for prediction in predictions
        ]
        formatted_predictions = [self.get_one_hot_vector(prediction) for prediction in split_predictions]
        result = self._metric.compute(
            predictions=formatted_predictions, references=formatted_references, average=self.average
        )
        if isinstance(result["f1"], numpy.ndarray):
            from statistics import mean

            final_result = {self.main_score: mean(result["f1"])}
            for i, label in enumerate(labels):
                final_result["f1_" + label] = result["f1"][i]
        else:
            final_result = {self.main_score: result["f1"]}
        return final_result


class F1MicroMultiLabel(F1MultiLabel):
    main_score = "f1_micro"
    average = "micro"


class F1MacroMultiLabel(F1MultiLabel):
    main_score = "f1_macro"
    average = None


class Rouge(HuggingfaceMetric):
    metric_name = "rouge"
    main_score = "rougeL"
    scale = 1.0

    def compute(self, references, predictions):
        predictions = ["\n".join(nltk.sent_tokenize(prediction.strip())) for prediction in predictions]
        references = [["\n".join(nltk.sent_tokenize(r.strip())) for r in reference] for reference in references]
        return super().compute(references, predictions)


class Bleu(HuggingfaceMetric):
    metric_name = "bleu"
    main_score = "bleu"
    scale = 1.0
