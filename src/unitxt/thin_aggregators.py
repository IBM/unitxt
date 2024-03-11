from collections import Counter
from math import sqrt
from typing import Any, List, Optional

import numpy as np


class Aggregator:
    def __init__(self, covered_metrics: Optional[List[str]] = None):
        self.covered_metrics = covered_metrics
        pass

    def single_instance_score(self, references: List[Any], prediction: Any) -> float:
        pass

    def accumulate_instance_value(self, references: List[Any], prediction: Any):
        pass

    def compute_final_from_aggregated(self) -> Any:
        pass


class MeanAndVar(Aggregator):
    def __init__(self, covered_metrics: Optional[List[str]] = None):
        super().__init__(covered_metrics)
        self.num_of_elements = 0
        self.sum_of_elements = 0.0
        self.sum_of_squares_of_elements = 0.0

    def accumulate_instance_value(self, instance_value: Any):
        if instance_value is None:
            return  # is that OK?
        self.num_of_elements += 1
        self.sum_of_elements += instance_value
        self.sum_of_squares_of_elements += instance_value**2

    # return a pair of (mean, variance), computed over non-nan values
    # if no valid values: return (None, None)
    def compute_final_from_aggregated(self) -> Any:
        if self.num_of_elements < 1:
            return (None, None)
        mn = self.sum_of_elements / self.num_of_elements
        return (mn, self.sum_of_squares_of_elements / self.num_of_elements - (mn**2))


class F1AccMatt(Aggregator):
    def __init__(self, covered_metrics: Optional[List[str]] = None):
        super().__init__(covered_metrics)
        self.confusion_matrix = Counter()

    def single_instance_score(self, references: List[Any], prediction: Any) -> float:
        # from F1(GlobalMetric) in metricts.py
        assert (
            len(references) == 1
        ), "Only a single reference per prediction is allowed in F1/Acc metrics"
        return 1.0 if prediction == references[0] else 0.0

    def accumulate_instance_value(self, references: List[Any], prediction: Any):
        # instance value for F1AccMatt is a pair (ref, pred), both are same type. ref is references[0] if more than one reference
        self.confusion_matrix.update([(references[0], prediction)])

    def compute_final_from_aggregated(self, metric_name):
        if metric_name == "metrics.f1_micro":
            return self.compute_f1_micro_from_confusion_matrix("metrics.f1_micro")
        if metric_name == "metrics.f1_macro":
            return self.compute_f1_macro_from_confusion_matrix()
        if metric_name == "metrics.accuracy":
            return self.compute_f1_micro_from_confusion_matrix(
                "metrics.accuracy"
            )  # accuracy is same as f1_micro
        if metric_name == "metrics.matthews_correlation":
            return self.compute_matthews_correlation_coefficient_from_confusion_matrix()
        raise ValueError("should not be here - metric name is wrong")

    # TODO: we can implement here what huggingface promise but not do: matthew_macro, reporting one result per class
    # Unitxt does not implement either: to see, try:
    # task="tasks.classification.multi_label[metrics=[metrics.matthews_correlation]]"
    # in card reuters21578, and continue to test card.
    def compute_matthews_correlation_coefficient_from_confusion_matrix(self) -> Any:
        # following https://dwbi1.wordpress.com/2022/10/05/mcc-formula-for-multiclass-classification/   that follows scikit-learn
        classes = list({r for r, p in self.confusion_matrix.keys()})  # true classes
        predictions = list(
            {p for r, p in self.confusion_matrix.keys()}
        )  # predictions, could also be a super set or a subset of classes
        intersect = [r for r in classes if r in predictions]
        tk = {
            ref: sum(
                self.confusion_matrix[(r, p)]
                for r, p in self.confusion_matrix.keys()
                if r == ref
            )
            for ref in classes
        }
        tksquared = sum(tk[ref] * tk[ref] for ref in tk.keys())
        pk = {
            pred: sum(
                self.confusion_matrix[(r, p)]
                for r, p in self.confusion_matrix.keys()
                if p == pred
            )
            for pred in predictions
        }
        pksquared = sum(pk[pred] * pk[pred] for pred in pk.keys())

        c = sum(self.confusion_matrix[(r, r)] for r in classes)
        s = sum(self.confusion_matrix.values())
        nominator = s * c - sum(tk[c] * pk[c] for c in intersect)
        denominator = sqrt(float(s * s - pksquared)) * sqrt(float(s * s - tksquared))

        mcc = nominator / denominator if denominator != 0 else np.nan
        if not np.isnan(mcc):
            mcc = round(mcc, 2)
        return {"matthews_correlation": mcc}

    def compute_f1_micro_from_confusion_matrix(self, metric_name) -> Any:
        # e.g. from here: https://www.baeldung.com/cs/multi-class-f1-score
        # or from here: https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/
        overall_true_positive = sum(
            self.confusion_matrix[r, p]
            for r, p in self.confusion_matrix.keys()
            if r == p
        )
        over_all_numof_instances = sum(self.confusion_matrix.values())
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

    def compute_f1_macro_from_confusion_matrix(self) -> Any:
        # e.g. from here: https://www.baeldung.com/cs/multi-class-f1-score
        # or from here: https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/
        over_all_numof_instances = sum(self.confusion_matrix.values())
        if (
            over_all_numof_instances == 0
        ):  # can happen with our tricky resampling, for very short streams
            return {"f1_macro": np.nan}
        true_classes = list(
            {r for r, p in self.confusion_matrix.keys()}
        )  # true classes
        predictions = list(
            {p for r, p in self.confusion_matrix.keys()}
        )  # predictions, could also be a super set or a subset of classes
        intersect = [r for r in true_classes if r in predictions]
        precision = {
            pred: float(self.confusion_matrix[(pred, pred)])
            / float(
                sum(
                    self.confusion_matrix[(r, p)]
                    for r, p in self.confusion_matrix.keys()
                    if p == pred
                )
            )
            for pred in intersect
        }
        recall = {
            ref: float(self.confusion_matrix[(ref, ref)])
            / float(
                sum(
                    self.confusion_matrix[(r, p)]
                    for r, p in self.confusion_matrix.keys()
                    if r == ref
                )
            )
            for ref in intersect
        }

        f1 = {
            "f1_" + str(c): 2 * precision[c] * recall[c] / (precision[c] + recall[c])
            if (precision[c] + recall[c]) > 0
            else 0.0
            for c in intersect
        }
        # for classes that never showed in any prediction, we have recall = 0,
        # and for classes that only showed as predictions (string-perturbated of class name) we have precision == 0
        # at any rate, these deserve f1 = 0, to contribute to average..
        f1.update({"f1_" + str(c): 0 for c in predictions if c not in intersect})
        f1.update({"f1_" + str(c): 0 for c in true_classes if c not in intersect})
        our_f1_macro = sum(f1.values()) / float(
            len(f1)
        )  # un weighted average over the classes

        f1.update({"f1_macro": round(our_f1_macro, 2)})
        return f1
