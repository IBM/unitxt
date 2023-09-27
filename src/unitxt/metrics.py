import re
import string
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import field
from typing import Any, Dict, Generator, List, Optional, Tuple

import evaluate
import numpy

from .dataclass import InternalField
from .operator import (
    MultiStreamOperator,
    SingleStreamOperator,
    StreamingOperator,
    StreamInstanceOperator,
)
from .operators import CopyFields
from .stream import MultiStream, Stream


def abstract_factory():
    return {}


def abstract_field():
    return field(default_factory=abstract_factory)


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

            try:
                instance_score = self._compute([refs], [pred])
            except:
                instance_score = {"score": None, "score_name": self.main_score}

                if isinstance(self.main_score, str) and self.main_score is not None:
                    instance_score[self.main_score] = None

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
        result["score_name"] = self.main_score
        return result

    @abstractmethod
    def compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        pass


class BulkInstanceMetric(SingleStreamOperator, Metric):
    main_score: str
    reduction_map: Dict[str, List[str]]

    implemented_reductions: List[str] = field(default_factory=lambda: ["mean"])

    def process(self, stream: Stream, stream_name: str = None) -> Generator:
        global_score = {}
        instances = []

        # consume the stream
        references, predictions = map(list, zip(*[(instance["references"],
                                                   instance["prediction"])
                                                  for instance in stream]))

        # compute the metric over all refs and preds
        instance_scores = self.compute(references=references, predictions=predictions)

        # add the score and score_name fields
        for instance_score in instance_scores:
            instance_score["score"] = instance_score[self.main_score]
            instance_score["score_name"] = self.main_score

        for instance, score in zip(stream, instance_scores):
            if "score" not in instance:
                instance["score"] = {"global": global_score, "instance": {}}
            else:
                global_score = instance["score"]["global"]

            instance["score"]["instance"].update(score)

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
                        global_score["score_name"] = self.main_score

        for instance in instances:
            yield instance

    @abstractmethod
    def compute(self, references: List[List[Any]], predictions: List[Any]) -> Dict[str, Any]:
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
                        global_score["score_name"] = self.main_score

        for instance in instances:
            yield instance

    def _compute(self, references: List[str], prediction: str) -> dict:
        result = self.compute(references=references, prediction=prediction)
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result

    @abstractmethod
    def compute(self, references: List[str], prediction: str) -> dict:
        pass


class Squad(GlobalMetric):
    _metric = None
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
        result["score_name"] = self.main_score
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
    hf_compute_args: dict = {}

    def prepare(self):
        super().prepare()
        self.metric = evaluate.load(self.metric_name)

    def compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        result = self.metric.compute(predictions=predictions, references=references, **self.hf_compute_args)
        if self.scale != 1.0:
            for key in result:
                if isinstance(result[key], float):
                    result[key] /= self.scale
        return result


class HuggingfaceBulkMetric(BulkInstanceMetric):

    metric_name: str

    hf_metric_fields: List[str]
    hf_compute_args: dict = {}

    def prepare(self):
        super().prepare()
        self.metric = evaluate.load(self.metric_name)

    def compute(self, references: List[List[str]], predictions: List[str]) -> List[Dict[str, Any]]:
        scores = self.metric.compute(predictions=predictions, references=references, **self.hf_compute_args)

        # convert dict of lists to a list of dicts
        results = [{} for _ in range(len(scores[self.hf_metric_fields[0]]))]
        for key in self.hf_metric_fields:
            values = scores[key]
            for result_id, result in enumerate(results):
                result[key] = values[result_id]

        return results


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
        ), "Only a single reference per prediction is allowed in F1 metric"
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
    classes_to_ignore = ["none"]

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
        assert all(
            len(reference) == 1 for reference in references
        ), "Only a single reference per prediction is allowed in F1 metric"
        references = [reference[0] for reference in references]
        labels = [
            l
            for l in set([label for reference in references for label in reference])
            if l not in self.classes_to_ignore
        ]
        # if no classes are left then F1 is not defined
        # (e.g. only "none" in references)
        if len(labels) == 0:
            return {self.main_score: float("nan")}

        for label in labels:
            self.add_str_to_id(label)
        formatted_references = [self.get_one_hot_vector(reference) for reference in references]
        formatted_predictions = [self.get_one_hot_vector(prediction) for prediction in predictions]

        # There is odd behavior in scikit-learn that when passing a one-hot vector with a single
        # element, it is treated a class identifier. Therefore, we add labels=[1] to limit to only
        # to this class.
        if len(labels) == 1:
            labels_param = [1]
        else:
            labels_param = None

        result = self._metric.compute(
            predictions=formatted_predictions,
            references=formatted_references,
            average=self.average,
            labels=labels_param,
        )
        if isinstance(result["f1"], numpy.ndarray):
            from statistics import mean

            assert len(result["f1"]) == len(
                labels
            ), f'F1 result ({result["f1"]}) has more entries than labels ({labels})'
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

    use_aggregator: bool = True
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    sent_split_newline: bool = True

    def prepare(self):
        self.hf_compute_args = {"use_aggregator": self.use_aggregator, "rouge_types": self.rouge_types}

        super().prepare()
        import nltk

        nltk.download("punkt")
        self.sent_tokenize = nltk.sent_tokenize

    def compute(self, references, predictions):
        if self.sent_split_newline:
            predictions = ["\n".join(self.sent_tokenize(prediction.strip())) for prediction in predictions]
            references = [["\n".join(self.sent_tokenize(r.strip())) for r in reference] for reference in references]
        return super().compute(references, predictions)


# Computes chat edit distance, ignoring whitespace
class CharEditDistanceAccuracy(SingleReferenceInstanceMetric):
    reduction_map = {"mean": ["char_edit_dist_accuracy"]}
    main_score = "char_edit_dist_accuracy"

    def prepare(self):
        import editdistance

        self.eval = editdistance.eval

    def compute(self, reference, prediction: str) -> dict:
        formatted_prediction = "".join(prediction.split())
        formatted_reference = "".join(reference.split())
        max_length = max(len(formatted_reference), len(formatted_prediction))
        if max_length == 0:
            return 0
        edit_dist = self.eval(formatted_reference, formatted_prediction)
        return {"char_edit_dist_accuracy": (1 - edit_dist / max_length)}


class Wer(HuggingfaceMetric):
    metric_name = "wer"
    main_score = "wer"

    def prepare(self):
        super().prepare()
        self.metric = evaluate.load(self.metric_name)

    def compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        assert all(
            len(reference) == 1 for reference in references
        ), "Only single reference per prediction is allowed in wer metric"
        formatted_references = [reference[0] for reference in references]
        result = self.metric.compute(predictions=predictions, references=formatted_references)
        return {self.main_score: result}


class Bleu(HuggingfaceMetric):
    metric_name = "bleu"
    main_score = "bleu"
    scale = 1.0


class SacreBleu(HuggingfaceMetric):
    metric_name = "sacrebleu"
    main_score = "score"
    scale = 1.0


class MatthewsCorrelation(HuggingfaceMetric):
    metric_name = "matthews_correlation"
    main_score = "matthews_correlation"
    str_to_id: dict = InternalField(default_factory=dict)

    def get_str_id(self, str):
        if str not in self.str_to_id:
            id = len(self.str_to_id)
            self.str_to_id[str] = id
        return self.str_to_id[str]

    def compute(self, references: List[List[str]], predictions: List[str]) -> dict:
        formatted_references = [self.get_str_id(reference[0]) for reference in references]
        formatted_predictions = [self.get_str_id(prediction) for prediction in predictions]
        result = self.metric.compute(predictions=formatted_predictions, references=formatted_references)
        return result


class CustomF1(GlobalMetric):
    main_score = "f1_micro"
    classes = None

    @abstractmethod
    def get_element_group(self, element):
        pass

    @abstractmethod
    def get_element_representation(self, element):
        pass

    def group_elements(self, l):
        return {
            k: Counter([self.get_element_representation(value) for value in l if self.get_element_group(value) == k])
            for k in set([self.get_element_group(e) for e in l])
        }

    def calculate_groups_ratio(self, actual_group, total_group):
        return sum([min(actual_group[k], total_group[k]) for k in actual_group.keys()]), sum(actual_group.values())

    def f1(self, pn, pd, rn, rd):
        precision = 1.0 if pn == 0 and pd == 0 else pn / pd
        recall = 1.0 if rn == 0 and rd == 0 else rn / rd
        try:
            return 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            return 0.0

    def compute(self, references: List[Any], predictions: List[Any]) -> dict:
        # in case reference are List[List[List[Any]]] and predictions are List[List[Any]]:
        if isinstance(references[0], list) and isinstance(references[0][0], list):
            references = [element[0] for element in references]

        assert len(references) == len(predictions), (
            f"references size ({len(references)})" f" doesn't mach predictions sise ({len(references)})."
        )
        if self.classes is None:
            classes = set([self.get_element_group(e) for sublist in references for e in sublist])
        else:
            classes = self.classes
        groups_statistics = dict()
        for references_batch, predictions_batch in zip(references, predictions):
            grouped_references = self.group_elements(references_batch)
            grouped_predictions = self.group_elements(predictions_batch)
            all_groups = set(grouped_references.keys()).union(grouped_predictions.keys())
            for group in all_groups:
                if group not in groups_statistics:
                    groups_statistics[group] = {
                        "precision_numerator": 0,
                        "precision_denominator": 0,
                        "recall_numerator": 0,
                        "recall_denominator": 0,
                    }
                references_by_group = grouped_references.get(group, Counter([]))
                predictions_by_group = grouped_predictions.get(group, Counter([]))
                pn, pd = self.calculate_groups_ratio(
                    actual_group=predictions_by_group, total_group=references_by_group
                )
                rn, rd = self.calculate_groups_ratio(
                    actual_group=references_by_group, total_group=predictions_by_group
                )
                groups_statistics[group]["precision_numerator"] += pn
                groups_statistics[group]["precision_denominator"] += pd
                groups_statistics[group]["recall_numerator"] += rn
                groups_statistics[group]["recall_denominator"] += rd

        result = {}
        num_of_unknown_class_predictions = 0
        pn_total = pd_total = rn_total = rd_total = 0
        for group in groups_statistics.keys():
            pn, pd, rn, rd = (
                groups_statistics[group]["precision_numerator"],
                groups_statistics[group]["precision_denominator"],
                groups_statistics[group]["recall_numerator"],
                groups_statistics[group]["recall_denominator"],
            )
            pn_total, pd_total, rn_total, rd_total = pn_total + pn, pd_total + pd, rn_total + rn, rd_total + rd
            if group in classes:
                result[f"f1_{group}"] = self.f1(pn, pd, rn, rd)
            else:
                num_of_unknown_class_predictions += pd
        try:
            result["f1_macro"] = sum(result.values()) / len(result.keys())
        except ZeroDivisionError:
            result["f1_macro"] = 1.0

        amount_of_predictions = pd_total
        if amount_of_predictions == 0:
            result["in_classes_support"] = 1.0
        else:
            result["in_classes_support"] = 1.0 - num_of_unknown_class_predictions / amount_of_predictions
        result[f"f1_micro"] = self.f1(pn_total, pd_total, rn_total, rd_total)
        return result


class NER(CustomF1):
    def get_element_group(self, element):
        return element[1]

    def get_element_representation(self, element):
        return str(element)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class TokenOverlap(InstanceMetric):
    reduction_map = {"mean": ["f1", "precision", "recall"]}
    main_score = "f1"

    def compute(self, references: List[Any], prediction: Any) -> dict:
        results = [self._compute_single_ref(reference, prediction)
                   for reference in references]
        return {measure: max(r[i] for r in results)
                for i, measure in enumerate(['precision', 'recall', 'f1'])}

    def _compute_single_ref(self, reference: Any, prediction: Any) -> Tuple[float, float, float]:
        prediction_tokens = normalize_answer(prediction).split()
        reference_tokens = normalize_answer(reference).split()
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            pr, rc, f1 = 0, 0, 0
        else:
            pr = 1.0 * num_same / len(prediction_tokens)
            rc = 1.0 * num_same / len(reference_tokens)
            f1 = (2 * pr * rc) / (pr + rc)
        return pr, rc, f1


class BertScore(HuggingfaceBulkMetric):
    metric_name = "bertscore"
    main_score = "f1"
    reduction_map = {"mean": ["f1", 'precision', 'recall']}
    hf_metric_fields = ["f1", "precision", "recall"]
    model_name: str

    def prepare(self):
        super().prepare()
        self.hf_compute_args = {"model_type": self.model_name}


class SentenceBert(BulkInstanceMetric):
    reduction_map = {"mean": ["score"]}
    main_score = "score"
    batch_size: int = 32

    model_name: str

    def prepare(self):
        super().prepare()
        from sentence_transformers import SentenceTransformer, util as sbert_util
        self.model = SentenceTransformer(self.model_name)
        self.util = sbert_util

    def compute(self, references: List[List[Any]], predictions: List[Any]) -> List[Any]:
        scores = []

        # we are in a multi-reference case (each prediction may have multiple
        # references), so we need to flatten the refs in order to compute the
        # embeddings in one batch, but first we have to store the spans of
        # reference groups, so we can recover it later on.
        ref_group_boundaries = []
        count = 0
        for ref_group in references:
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

        # compute s-bert embeddings
        preds_emb = self.model.encode(predictions)
        refs_emb = self.model.encode([ref for ref_group in references for ref in ref_group])

        # for each candidate, pick the reference with the highest score
        for pred_emb, ref_group_bounds in zip(preds_emb, ref_group_boundaries):
            refs_group_emb = refs_emb[ref_group_bounds[0]:ref_group_bounds[1]]
            scores.append(self.util.cos_sim(pred_emb, refs_group_emb).max().item())

        return [{"score": score} for score in scores]


class Reward(BulkInstanceMetric):
    reduction_map = {"mean": ["score"]}
    main_score = "score"
    batch_size: int = 32

    model_name: str

    def prepare(self):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=self.model_name)

    def compute(self, references: List[List[Any]], predictions: List[Any]) -> List[Any]:
        # treat the references as the questions and the predictions as answers
        # assume a single reference
        questions = [refs[0] for refs in references]
        answers = predictions

        # prepare for computation
        inputs = [{"text": q, "text_pair": a}
                  for q, a in zip(questions, answers)]

        # compute the metric
        # add function_to_apply="none" to disable sigmoid
        return self.pipe(inputs, batch_size=self.batch_size)
        # return [r['score'] for r in self.pipe(inputs, batch_size=self.batch_size)]

