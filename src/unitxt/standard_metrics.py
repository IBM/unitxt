from collections import Counter, defaultdict
from math import isclose
from typing import Any, Dict, List

import numpy as np

from .artifact import Artifact, fetch_artifact
from .type_utils import isoftype


class MetricFetcherMixin:
    """Provides a way to fetch and cache artifacts in the system.

    Args:
        cache (Dict[str, Artifact]): A cache for storing fetched artifacts.
    """

    cache: Dict[str, Artifact] = {}

    @classmethod
    def get_artifact(cls, artifact_identifier: str) -> Artifact:
        if artifact_identifier not in cls.cache:
            artifact, artifactory = fetch_artifact(artifact_identifier)
            cls.cache[artifact_identifier] = artifact
        return cls.cache[artifact_identifier]


class Aggregator(Artifact, MetricFetcherMixin):
    metric_name: str

    def prepare(self):
        pass

    def single_instance_score(self, references: List[Any], prediction: Any):
        pass

    def accumulate_instance_value(self, references: List[Any], prediction: Any):
        pass

    def compute_final_from_aggregated(self) -> dict:
        pass


class ConfusionMatrixAggregator(Aggregator):
    def prepare(self):
        self.confusion_matrix = Counter()

    def single_instance_score(self, references: List[Any], prediction: Any) -> float:
        # from F1(GlobalMetric) in metricts.py
        # assert (
        #     len(references) == 1
        # ), f"Only a single reference per prediction is allowed for {self.metric_name}"
        return {self.metric_name[8:]: 1.0 if prediction in references else 0.0}

    def accumulate_instance_value(self, references: List[Any], prediction: Any):
        # from F1(GlobalMetric) in metricts.py
        # assert (
        #     len(references) == 1
        # ), f"Only a single reference per prediction is allowed for {self.metric_name}"
        self.confusion_matrix.update([(references[0], prediction)])


def compute_acc_f1_micro_from_confusion_matrix(
    confusion_matrix: Counter, metric_name: str
) -> Any:
    # e.g. from here: https://www.baeldung.com/cs/multi-class-f1-score
    # or from here: https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/
    overall_true_positive = sum(
        confusion_matrix[r, p] for r, p in confusion_matrix.keys() if r == p
    )
    over_all_numof_instances = sum(confusion_matrix.values())
    if (
        over_all_numof_instances == 0
    ):  # can happen with our tricky resampling, for very short streams
        return {metric_name[8:]: np.nan}
    precision = float(overall_true_positive) / float(over_all_numof_instances)
    # for micro, overall_precision == overall_recall, we thus have:
    # our_micro_f1 = 2 * precision * precision / (precision + precision) = precision
    # as nicely explained here:
    # https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/
    our_f1_micro = precision
    return {metric_name[8:]: round(our_f1_micro, 2)}


def compute_f1_macro_from_confusion_matrix(confusion_matrix: Counter) -> Any:
    # e.g. from here: https://www.baeldung.com/cs/multi-class-f1-score
    # or from here: https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/
    over_all_numof_instances = sum(confusion_matrix.values())
    if (
        over_all_numof_instances == 0
    ):  # can happen with our tricky resampling, for very short streams
        return {"f1_macro": np.nan}
    true_classes = list({r for r, p in confusion_matrix.keys()})  # true classes
    predictions = list(
        {p for r, p in confusion_matrix.keys()}
    )  # predictions, could also be a super set or a subset of classes
    intersect = [r for r in true_classes if r in predictions]
    precision = {
        pred: float(confusion_matrix[(pred, pred)])
        / float(
            sum(
                confusion_matrix[(r, p)]
                for r, p in confusion_matrix.keys()
                if p == pred
            )
        )
        for pred in intersect
    }
    recall = {
        ref: float(confusion_matrix[(ref, ref)])
        / float(
            sum(
                confusion_matrix[(r, p)] for r, p in confusion_matrix.keys() if r == ref
            )
        )
        for ref in intersect
    }

    f1 = {
        "f1_" + str(c): round(
            2 * precision[c] * recall[c] / (precision[c] + recall[c]), 2
        )
        if (precision[c] + recall[c]) > 0
        else 0.0
        for c in intersect
    }
    # for classes that never showed in any prediction, we have recall = 0,
    # and for classes that only showed as predictions (string-perturbated of class name) we have precision == 0
    # at any rate, these deserve f1 = 0, to contribute to average..
    f1.update({"f1_" + str(c): 0.0 for c in predictions if c not in intersect})
    f1.update({"f1_" + str(c): 0.0 for c in true_classes if c not in intersect})
    our_f1_macro = sum(f1.values()) / float(
        len(f1)
    )  # un weighted average over the classes

    f1.update({"f1_macro": round(our_f1_macro, 2)})
    return f1


class AccuracyF1Aggregator(ConfusionMatrixAggregator):
    def compute_final_from_aggregated(self) -> dict:
        return compute_acc_f1_micro_from_confusion_matrix(
            self.confusion_matrix, self.metric_name
        )


class F1MacroAggregator(ConfusionMatrixAggregator):
    def compute_final_from_aggregated(self) -> dict:
        return compute_f1_macro_from_confusion_matrix(self.confusion_matrix)


class ConfusionMatrixForMultiLabelAggregator(Aggregator):
    def prepare(self):
        self.classes_seen_thus_far = set()
        self.references_seen_thus_far = set()
        self.tp = defaultdict(int)  # true/false positive/negative of what seen thus far
        self.tn = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.num_of_instances_seen_thus_far = 0

    def single_instance_score(self, references: List[Any], prediction: Any) -> float:
        self._validate_references_and_prediction(
            references=references, prediction=prediction
        )
        num_of_hits = len([pred for pred in prediction if pred in references[0]])
        # this method is invoked for storing instance[score][instance], after global was computed,
        # so self.classes_seen_thus_far is updated with all classes ever seen in pred or ref
        # for coherence, and in order to clean results, we only report on classes ever seen in references
        # num_of_true_misses = len(
        #     [
        #         ref
        #         for ref in self.references_seen_thus_far
        #         if ref not in prediction and ref not in references[0]
        #     ]
        # )
        # hit_ratio = float(num_of_hits + num_of_true_misses) / len(prediction)
        hit_ratio = 0.0 if len(prediction) == 0 else num_of_hits / len(prediction)
        return {self.metric_name[8:]: round(hit_ratio, 2)}

    def accumulate_instance_value(self, references: List[Any], prediction: Any):
        self._validate_references_and_prediction(
            references=references, prediction=prediction
        )
        # two lists of distinct str (distinct inside each list), reference is wrapped.
        # we increase tp for each member of pred that is also in ref.
        # we increase fp for each member of pred that is not in ref,
        # we increase fn for each member of ref that is not in pred.
        # we increase tn for all members of classes that we know of, that are missing from both,
        #
        # once a new class becomes known to us, we increase its tn by the number of instances seen thus
        # far (not including this one).
        for pred in prediction:
            if pred in references[0]:
                self.tp[pred] += 1
            else:
                self.fp[pred] += 1
        for ref in references[0]:
            self.references_seen_thus_far.add(ref)
            if ref not in prediction:
                self.fn[ref] += 1
        for c in self.classes_seen_thus_far:
            if c not in prediction and c not in references[0]:
                self.tn[c] += 1
        for d in prediction + references[0]:
            if d not in self.classes_seen_thus_far:
                self.tn[d] = self.num_of_instances_seen_thus_far
                self.classes_seen_thus_far.add(d)
        self.num_of_instances_seen_thus_far += 1

    def compute_final_from_aggregated(self) -> dict:
        if self.metric_name in ["metrics.f1_micro_multi_label", "metrics.accuracy"]:
            return (
                self.compute_accuracy_and_f1_micro_multi_label_from_confusion_matrix()
            )
        if self.metric_name == "metrics.f1_macro_multi_label":
            return self.compute_f1_macro_multi_label_from_confusion_matrix()
        # if metric_name == "metrics.matthews_correlation":
        #     return self.compute_matthews_correlation_coefficient_from_confusion_matrix()
        raise ValueError("should not be here - metric name is wrong")

    def compute_accuracy_and_f1_micro_multi_label_from_confusion_matrix(self):
        if self.num_of_instances_seen_thus_far == 0:
            return {self.metric_name[8:]: np.nan}
        total_tp = sum(v for k, v in self.tp.items())
        total_tn = sum(v for k, v in self.tn.items())
        total_fp = sum(v for k, v in self.fp.items())
        total_fn = sum(v for k, v in self.fn.items())
        if self.metric_name == "metrics.accuracy":
            acc = (total_tp + total_tn) / self.num_of_instances_seen_thus_far
            return {"accuracy": round(acc, 2)}
        if self.metric_name == "metrics.f1_micro_multi_label":
            if total_tp == 0:
                return {"f1_micro_multi_label": 0.0}
            total_recall = total_tp / (total_fn + total_tp)
            total_precision = total_tp / (total_tp + total_fp)
            f1_micro = (
                2 * total_recall * total_precision / (total_recall + total_precision)
            )
            return {"f1_micro_multi_label": round(f1_micro, 2)}
        return None

    # adapted from metrics.py for multilabel
    def _validate_references_and_prediction(
        self, references: List[Any], prediction: Any
    ):
        if not len(references) == 1:
            raise ValueError(
                f"Only a single reference per instance is allowed in multi label metric. Received reference: {references}"
            )
        if not isoftype(references[0], List[str]):
            raise ValueError(
                f"Instance references is expected to be a list of one item being a list of strings in multi label metric. Received references: '{references}'"
            )

        if not isoftype(prediction, List[str]):
            raise ValueError(
                f"Instance prediction is expected to be a list of strings in multi label metric. Received prediction: '{prediction}'"
            )
        if not len(set(prediction)) == len(prediction):
            raise ValueError(
                f"Elements of prediction are expected to be distinct strings, in multi label metric. Received prediction: '{prediction}'"
            )
        if not len(set(references[0])) == len(references[0]):
            raise ValueError(
                f"Elements of references[0] are expected to be distinct strings, in multi label metric. Received references[0]: '{references[0]}'"
            )

    def compute_f1_macro_multi_label_from_confusion_matrix(self) -> Any:
        # e.g. from here: https://medium.com/synthesio-engineering/precision-accuracy-and-f1-score-for-multi-label-classification-34ac6bdfb404
        # report only for the classes seen as references
        if len(self.references_seen_thus_far) == 0:
            return {self.metric_name[8:]: np.nan}
        to_ret = {}
        for c in self.references_seen_thus_far:  # report only on them
            num_as_pred = self.tp[c] + self.fp[c]
            precision = 0.0 if num_as_pred == 0 else self.tp[c] / num_as_pred
            num_as_ref = self.tp[c] + self.fn[c]
            recall = np.nan if num_as_ref == 0 else self.tp[c] / num_as_ref
            f1 = (
                0.0
                if np.isnan(precision)
                or np.isnan(recall)
                or isclose(precision, 0)
                or isclose(recall, 0)
                else 2 * precision * recall / (precision + recall)
            )
            to_ret["f1_" + c] = round(f1, 2)
        avg_across_classes = (
            sum(val for val in to_ret.values() if not np.isnan(val))
        ) / len(self.references_seen_thus_far)
        to_ret[self.metric_name[8:]] = round(avg_across_classes, 2)
        return to_ret


class AccF1MultiLabelAggregator(ConfusionMatrixForMultiLabelAggregator):
    pass
