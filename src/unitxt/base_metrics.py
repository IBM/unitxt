import warnings
from abc import abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
from scipy.stats._warnings_errors import DegenerateDataWarning

from .aggregators import Aggregator, MeanAggregator
from .artifact import Artifact
from .dataclass import AbstractField, Field, OptionalField
from .logging_utils import get_logger
from .operator import StreamOperator
from .settings_utils import get_settings
from .stream import Stream
from .type_utils import Type, isoftype, parse_type_string, to_type_string

logger = get_logger()
settings = get_settings()
warnings.filterwarnings("ignore", category=DegenerateDataWarning)


def parse_string_types_instead_of_actual_objects(obj):
    return parse_type_string(obj)


class Metric(Artifact):
    main_score: str = AbstractField()
    # Override 'prediction_type' with the expected type of predictions
    # and references.  Example: "List[str]", "List[Dict]"", "string".
    # If left with default None, a warning will be displayed.
    # In future versions of unitxt, this will be an error.
    prediction_type: Union[Type, str] = Any

    reference_field: str = "references"
    prediction_field: str = "prediction"
    task_data_field: str = "task_data"

    # Standard metrics can receive multiple references per predictions (in a list)
    # Some metrics support only a single reference per prediction (one element in the list)
    single_reference_per_prediction: bool = False

    # Used to store the parsed prediction type and avoid
    # parsing on every use
    _parsed_prediction_type = None

    #
    # Used to add a prefix to all score, except the "score_name" and "score" fields.
    # This is used to distinguish two scores of the same metrics, operating on different fields of the task
    #
    score_prefix: str = ""

    def prepare(self):
        super().prepare()
        if isinstance(self.prediction_type, str):
            self.prediction_type = parse_string_types_instead_of_actual_objects(
                self.prediction_type
            )

    @classmethod
    def process_data_after_load(cls, data):
        if "prediction_type" in data:
            data["prediction_type"] = parse_type_string(data["prediction_type"])
        return data

    def process_data_before_dump(self, data):
        if "prediction_type" in data:
            if not isinstance(data["prediction_type"], str):
                data["prediction_type"] = to_type_string(data["prediction_type"])
        return data

    def _add_score_prefix(self, score_name):
        return (
            self.score_prefix + score_name
            if score_name not in ["score", "score_name"]
            else score_name
        )

    def _add_score_prefixes_to_score_dict(self, scores: Dict[str, Any]):
        new_scores = {}
        for score_name, score in scores.items():
            score_with_prefix = self._add_score_prefix(score_name)
            new_scores[score_with_prefix] = (
                score if score_name not in ["score_name"] else self.score_prefix + score
            )
        return new_scores

    # for instance scores that only include the raw score names
    def add_score_score_name_and_score_prefix_to_instance_score(
        self, instances: List[Dict[str, Any]]
    ):
        for instance in instances:
            if self.main_score not in instance["score"]["instance"]:
                # possible for global metric, for example, that do not score every instance
                continue

            if len(self.score_prefix) == 0:
                instance["score"]["instance"]["score_name"] = self.main_score
                instance["score"]["instance"]["score"] = instance["score"]["instance"][
                    self.main_score
                ]

            else:
                new_instance_score = {}
                new_instance_score["score_name"] = self.score_prefix + self.main_score
                new_instance_score["score"] = instance["score"]["instance"][
                    self.main_score
                ]

                for score_name, score in instance["score"]["instance"].items():
                    if score_name in ["score", "score_name"]:
                        # already in new_instance_score
                        continue
                    if score_name not in self.recently_added_instance_scores:
                        # copy as is:
                        new_instance_score[score_name] = score
                    else:
                        new_instance_score[self.score_prefix + score_name] = score
                instance["score"]["instance"] = new_instance_score

    def _validate_references_and_predictions(self, references, predictions):
        if not isoftype(predictions, List[Any]):
            raise ValueError(
                f"Metric {self.get_metric_name()} should receive a list of predictions {self.get_metric_name()}.  Received predictions of type {type(predictions)}: {predictions}"
            )

        if not isoftype(references, List[Any]):
            raise ValueError(
                f"Metric {self.get_metric_name()} should receive a list of predictions. Received references of type {type(references)}: {references}"
            )

        if len(references) != len(predictions):
            raise ValueError(
                f"references size ({len(references)})"
                f" doesn't mach predictions size ({len(references)})."
            )

        for reference in references:
            self._validate_reference(reference)

        for prediction in predictions:
            self._validate_prediction(prediction)

    def _validate_prediction(self, prediction):
        if not isoftype(prediction, self.prediction_type):
            raise ValueError(
                f"Each prediction is expected to be of type '{to_type_string(self.prediction_type)}' in {self.get_metric_name()} metric. Received prediction of type {type(prediction)}: {prediction}"
            )

    def _validate_reference(self, reference):
        if not isoftype(reference, List[Any]):
            raise ValueError(
                f"Expecting a list of references for each prediction in {self.get_metric_name()} metric. Received reference of type {type(reference)}: {reference}"
            )
        if self.single_reference_per_prediction and not len(reference) == 1:
            raise ValueError(
                f"Expecting a list with a single reference per prediction in {self.get_metric_name()} metric. Received a list with multiple references: {reference}"
            )
        for ref in reference:
            if not isoftype(ref, self.prediction_type):
                raise ValueError(
                    f"Each reference is expected to be of type '{to_type_string(self.prediction_type)}' in {self.get_metric_name()} metric. Received reference of type {type(ref)}: {ref}"
                )

    def get_metric_name(self):
        if self.__id__ is not None:
            return self.__id__
        return self.__class__.__name__

    @abstractmethod
    def disable_confidence_interval_calculation(self):
        pass

    # needed also be RemoteMetric
    def consume_stream(
        self,
        stream: Stream,
    ):
        references_list = []
        prediction_list = []
        task_data_list = []
        instance_list = []
        for instance in stream:
            instance = self.verify_instance(instance)

            task_data = (
                instance[self.task_data_field]
                if self.task_data_field in instance
                else {}
            )

            if self.reference_field == "references":
                refs = instance["references"]
            else:
                refs = task_data[self.reference_field]
                if not isinstance(refs, list):
                    refs = [refs]
            if self.prediction_field == "prediction":
                pred = instance["prediction"]
            else:
                pred = task_data[self.prediction_field]

            references_list.append(refs)
            prediction_list.append(pred)
            task_data_list.append(task_data)

            if "score" not in instance:
                instance["score"] = {"global": {}, "instance": {}}
            if "instance" not in instance["score"]:
                instance["score"]["instance"] = {}
            if "global" not in instance["score"]:
                instance["score"]["global"] = {}
            instance_list.append(instance)
        self._validate_references_and_predictions(references_list, prediction_list)
        return prediction_list, references_list, task_data_list, instance_list


class MetricWithConfidenceInterval(Metric, StreamOperator):
    # The number of resamples used to estimate the confidence intervals of this metric.
    # From all over unitxt, use None to disable confidence interval computation.
    n_resamples: int = None
    ci_scores: List[str] = None

    # aggregates along the instances, according to all settings
    aggregator: Aggregator = None
    score_names: List[str] = None

    # the name to associate with the aggregator, to participate in prefixes for score_names
    aggregating_function_name: str = ""

    @abstractmethod
    def compute_each_single_instance(
        self,
        references_list: List[List[Any]],
        prediction_list: List[Any],
        task_data_list: List[Dict[str, Any]],
        instance_list: List[Dict[str, Any]],
    ):
        pass
        # also updates instance["score"]["instance"] but not yet prefixed

    def compute_all_instances(
        self,
        references_list: List[List[Any]],
        prediction_list: List[Any],
        task_data_list: List[Dict[str, Any]],
        instance_list: List[Dict[str, Any]],
    ):
        # We now proceed to calculate global score, still no prefixes and no "score" and "score_name"
        self.aggregator.set_metric_related_properties(
            is_serving_global_metric=isinstance(self, GlobalMetric),
            n_resamples=self.n_resamples,
            ci_scores=self.ci_scores,
            main_score=self.main_score,
            score_prefix=self.score_prefix,
            aggregating_function_name=self.aggregating_function_name,
            confidence_level=0.95,
        )

        # do everything related to global score and CI
        # also updates the global scores in the instance["score"]["global"] -s
        # still all score names are non-prefixed
        self.aggregator.compute_global_score(instance_list)

    def prepare(self):
        assert self.aggregator is not None
        assert issubclass(type(self.aggregator), Aggregator)
        if not hasattr(self, "score_names") or self.score_names is None:
            self.score_names = [self.main_score]
        if self.aggregator.score_names is None:
            self.aggregator.score_names = self.score_names
        if self.ci_scores is None:
            self.ci_scores = [self.main_score]
        self.recently_added_instance_scores = [self.main_score]
        super().prepare()

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        (
            prediction_list,
            references_list,
            task_data_list,
            instance_list,
        ) = self.consume_stream(stream=stream)

        self.compute_each_single_instance(
            references_list, prediction_list, task_data_list, instance_list
        )

        # the following includes the update of global scores in the instances
        self.compute_all_instances(
            references_list, prediction_list, task_data_list, instance_list
        )

        # these, and the instance_scores were kept in their original name (unprefixed) to ease aggregation overthem
        # so now we complete all these prefixes
        self.add_score_score_name_and_score_prefix_to_instance_score(instance_list)

        yield from instance_list

    def disable_confidence_interval_calculation(self):
        self.n_resamples = None


class GlobalMetric(MetricWithConfidenceInterval, Aggregator):
    """A class for computing metrics that require joint calculations over all instances and are not just aggregation of scores of individuals instances.

    For example, macro_F1 requires
    calculation requires calculation of recall and precision per class, so all instances of the class
    need to be considered.  Accuracy, on the other hand, is just an average of the accuracy of all the instances.
    """

    n_resamples = OptionalField(
        default_factory=lambda: settings.num_resamples_for_global_metrics
    )

    # calculate scores for single instances
    process_single_instances = True

    aggregating_function_name = ""
    # for global metric, the bare score names carry the name of the aggregation

    def prepare(self):
        if self.aggregator is None:
            self.aggregator = self
        self.score_names = [self.main_score]
        super().prepare()

    def compute_each_single_instance(
        self,
        references_list: List[List[Any]],
        prediction_list: List[Any],
        task_data_list: List[Dict[str, Any]],
        instance_list: List[Dict[str, Any]],
    ):
        for prediction, references, task_data, instance in zip(
            prediction_list, references_list, task_data_list, instance_list
        ):
            instance_score = None

            # for backward compatibility
            no_score_value = np.nan
            if self.process_single_instances:
                try:
                    instance_score = self.compute(
                        [references], [prediction], [task_data]
                    )
                except:
                    no_score_value = None
            if not instance_score:
                instance_score = {
                    "score": no_score_value,
                    "score_name": self.main_score,
                }

                if isinstance(self.main_score, str):
                    instance_score[self.main_score] = no_score_value

            instance["score"]["instance"].update(instance_score)

    def aggregate(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        predictions, references, task_data, _ = self.consume_stream(stream=instances)
        return self.compute(references, predictions, task_data)

    @abstractmethod
    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Any],
    ) -> dict:
        """Computes a scores dictionary on a list of references, predictions and input.

        This function is called once per instance, and then another time
        over all data instances.

        Returns:
            a dictionary of scores that is set as:
              the instance scores when called on a single data instance
              the global score when called on the all data instances
        """
        pass


class BulkInstanceMetric(MetricWithConfidenceInterval):
    n_resamples = OptionalField(
        default_factory=lambda: settings.num_resamples_for_instance_metrics
    )
    main_score: str
    score_names: List[str] = None

    aggregator = Field(default_factory=lambda: MeanAggregator(score_names=None))
    aggregating_function_name = "mean"

    def prepare(self):
        if self.main_score is None:
            self.main_score = "f1"
        super().prepare()
        self.ci_scores = list(set(self.ci_scores))

    # just to give implementations of the abstract methods of MetricWithConfidenceInterval
    # this class uses its own process()
    def compute_each_single_instance(
        self,
        references_list: List[List[Any]],
        prediction_list: List[Any],
        task_data_list: List[Dict[str, Any]],
        instance_list: List[Dict[str, Any]],
    ):
        # compute the metric over all refs and preds
        instance_scores = self.compute(
            references=references_list,
            predictions=prediction_list,
            task_data=task_data_list,
        )
        for instance, score in zip(instance_list, instance_scores):
            instance["score"]["instance"].update(score)
        # "score_name", and "score"  and prefixes are not yet updated in instance scores

    @abstractmethod
    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        pass


class InstanceMetric(MetricWithConfidenceInterval):
    """Class for metrics for which a global score can be calculated by aggregating the instance scores (possibly with additional instance inputs)."""

    n_resamples = OptionalField(
        default_factory=lambda: settings.num_resamples_for_instance_metrics
    )

    aggregator = Field(default_factory=lambda: MeanAggregator(score_names=None))
    aggregating_function_name = "mean"

    def compute_each_single_instance(
        self,
        references_list: List[List[Any]],
        prediction_list: List[Any],
        task_data_list: List[Dict[str, Any]],
        instance_list: List[Dict[str, Any]],
    ):
        for prediction, references, task_data, instance in zip(
            prediction_list, references_list, task_data_list, instance_list
        ):
            instance_score = self.compute(
                references=references, prediction=prediction, task_data=task_data
            )
            self.recently_added_instance_scores = list(instance_score.keys())
            instance["score"]["instance"].update(instance_score)

    @abstractmethod
    def compute(self, references: List[Any], prediction: Any, task_data: Dict) -> dict:
        pass
