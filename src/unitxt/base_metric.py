from abc import abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Union,
)

from .artifact import Artifact
from .dataclass import (
    AbstractField,
)
from .deprecation_utils import deprecation
from .error_utils import Documentation, UnitxtWarning
from .stream import Stream
from .type_utils import Type, isoftype, parse_type_string, to_type_string


@deprecation(
    version="2.0.0",
    msg="use regular type instead of strings (e.g Dict[str] instead of 'Dict[str]')",
)
def parse_string_types_instead_of_actual_objects(obj):
    return parse_type_string(obj)


class Metric(Artifact):
    main_score: str = AbstractField()
    # Override 'prediction_type' with the expected type of predictions
    # and references.  Example: "List[str]", "List[Dict]"", "string".
    # If left with default None, a warning will be displayed.
    # In future versions of unitxt, this will be an error.
    prediction_type: Union[Type, str] = Any

    # Standard metrics can receive multiple references per predictions (in a list)
    # Some metrics support only a single reference per prediction (one element in the list)
    single_reference_per_prediction: bool = False

    #
    # Used to add a prefix to all score, except the "score_name" and "score" fields.
    # This is used to distinguish two scores of the same metrics, operating on different fields of the task
    #
    score_prefix: str = ""

    def prepare_args(self):
        super().prepare_args()
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
            if score_name not in ["score", "score_name", "num_of_instances"]
            else score_name
        )

    def _add_score_prefixes_to_score_dict_and_check_against_existing_scores(
        self, scores: Dict[str, Any], existing_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        new_scores = {}
        for score_name, score in scores.items():
            score_with_prefix = self._add_score_prefix(score_name)
            new_scores[score_with_prefix] = (
                score if score_name not in ["score_name"] else self.score_prefix + score
            )
        for new_score_name in new_scores:
            if new_score_name in ["score", "score_name", "num_of_instances"]:
                continue
            if new_score_name in existing_scores:
                UnitxtWarning(
                    message=f"Metric '{new_score_name}' that has just been evaluated to {new_scores[new_score_name]}, is already recorded "
                    f"to have value {existing_scores[new_score_name]} by a previous metric evaluation on this instance or stream. "
                    f"To avoid overwriting the existing value, add a score_prefix to the metric name (e.g. score_prefix='my_second_' , "
                    f"which will yield, in this case, a score named: 'my_second_{new_score_name}')",
                    additional_info_id=Documentation.MULTIPLE_METRICS_OUTPUTS,
                )
        return new_scores

    def _validate_references_and_prediction(self, references, predictions):
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

    def consume_stream(self, stream: Stream):
        references = []
        predictions = []
        additional_inputs = []
        instances = []
        for instance in stream:
            instance = self.verify_instance(instance)
            references.append(instance["references"])
            predictions.append(instance["prediction"])
            additional_inputs.append(
                instance["additional_inputs"] if "additional_inputs" in instance else {}
            )
            instances.append(instance)
        return predictions, references, additional_inputs, instances

    @staticmethod
    def update_instance_scores(instances, instances_scores: List[Dict[str, Any]]):
        for instance, new_scores in zip(instances, instances_scores):
            if "score" not in instance:
                instance["score"] = {}
            scores = instance["score"]
            if "instance" not in scores:
                scores["instance"] = {}
            scores["instance"].update(new_scores)

    @staticmethod
    def set_global_score(instances, global_score: Dict[str, Any]):
        for instance in instances:
            if "score" not in instance:
                instance["score"] = {}
            scores = instance["score"]
            if "global" not in scores:
                scores["global"] = {}
            scores["global"] = global_score

    @abstractmethod
    def set_confidence_interval_calculation(self, return_confidence_interval: bool):
        pass

    def disable_confidence_interval_calculation(self):  # For backward compatibility
        self.set_confidence_interval_calculation(return_confidence_interval=False)

    # update instance["score"]["global"] with the global_score just computed for the
    # current metric.  global_score contains "score" and "score_name" fields that reflect
    # (the main_score of) the current metric. If CI was computed for global_score, then global_score
    # also contains "score_ci_low" and "score_ci_high" that reflect (the main_score of) the current metric.
    # A simple python-dictionary-update adds new fields to instance["score"]["global"], and also replaces the values
    # of its fields "score" and "score_name" (and "score_ci_low", "score_ci_high" if applicable),
    # to reflect the current metric, overwriting previous metrics' settings of these fields
    # (if any previous metric exists).
    # When global_score does NOT contain ci score (because CI was not computed for the current metric), but
    # one of the previous metrics computed did have, the last of such previous metrics set the values in
    # fields "score_ci_low" and "score_ci_high" in instance["score"]["global"] to reflect its
    # (the previous metric's) CI scores.
    # Because CI is not computed for the current metric, global_score does not contain fields "score_ci_low" and
    # "score_ci_high" to overwrite the ones existing in instance["score"]["global"], and these might remain in
    # instance["score"]["global"], but their values, that are not associated with the current metric, are,
    # therefore, not consistent with "score_name".
    # In such a case, following the python-dictionary-update, we pop out fields "score_ci_low" and
    # "score_ci_high" from instance["score"]["global"], so that now all the fields "score.." in
    # instance["score"]["global"] are consistent with the current metric: The metric that is named
    # instance["score"]["global"]["score_name"], its score shows in
    # field instance["score"]["global"]["score"], and it does not have ci_scores,
    # which is also reflected in the absence of fields "score_ci_low" and "score_ci_high" from instance["score"]["global"].
    # If ci IS computed for the current metric, global_score contains "score_ci_low" and "score_ci_high", and these overwrite
    # the ones existing in instance["score"]["global"] by the simple python-dictionary-update, and no need for any further fixeup.
    def update_and_adjust_global_score(
        self, instance: Dict[str, Any], global_score: dict
    ):
        for score_name in global_score:
            if score_name in [
                "score",
                "score_name",
                "score_ci_low",
                "score_ci_high",
                "num_of_instances",
            ]:
                continue
            if score_name in instance["score"]["global"]:
                UnitxtWarning(
                    message=f"Global metric '{score_name}' that has just been evaluated to {global_score[score_name]}, is already recorded "
                    f"to have value {instance['score']['global'][score_name]} by a previous metric evaluation on this stream. "
                    f"To avoid overwriting the value, add a score_prefix to the metric (e.g. score_prefix='my_{score_name}'.",
                    additional_info_id=Documentation.MULTIPLE_METRICS_OUTPUTS,
                )
        instance["score"]["global"].update(global_score)
        for score_ci in ["score_ci_low", "score_ci_high"]:
            if score_ci in global_score:
                continue
            if score_ci in instance["score"]["global"]:
                instance["score"]["global"].pop(score_ci)
