from collections import Counter
from typing import Any, Dict, List

import numpy as np

from .artifact import Artifact, fetch_artifact


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
        assert (
            len(references) == 1
        ), f"Only a single reference per prediction is allowed for {self.metric_name}"
        return {self.metric_name[8:]: 1.0 if prediction in references else 0.0}

    def accumulate_instance_value(self, references: List[Any], prediction: Any):
        # from F1(GlobalMetric) in metricts.py
        assert (
            len(references) == 1
        ), f"Only a single reference per prediction is allowed for {self.metric_name}"
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
