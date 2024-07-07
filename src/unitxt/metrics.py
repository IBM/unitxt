import ast
import re
import string
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import field
from operator import itemgetter
from statistics import mean
from typing import Any, Dict, Generator, List, Optional, Tuple

import evaluate
import numpy
import numpy as np
from scipy.stats import bootstrap
from scipy.stats._warnings_errors import DegenerateDataWarning

from .artifact import Artifact
from .dataclass import (
    AbstractField,
    InternalField,
    NonPositionalField,
    OptionalField,
)
from .inference import HFPipelineBasedInferenceEngine, InferenceEngine
from .logging_utils import get_logger
from .metric_utils import InstanceInput, MetricRequest, MetricResponse
from .operator import (
    InstanceOperator,
    MultiStreamOperator,
    SequentialOperator,
    StreamingOperator,
    StreamOperator,
)
from .operators import Copy
from .random_utils import get_seed
from .settings_utils import get_settings
from .stream import MultiStream, Stream
from .type_utils import isoftype, parse_type_string

logger = get_logger()
settings = get_settings()

warnings.filterwarnings("ignore", category=DegenerateDataWarning)


warnings.filterwarnings("ignore", category=DegenerateDataWarning)


def abstract_factory():
    return {}


def abstract_field():
    return field(default_factory=abstract_factory)


def nan_mean(x):
    with warnings.catch_warnings():
        # final mean should be mean of scores, ignoring NaN, hence nanmean
        # but if the group function values is NaN for ALL values, nanmean throws a
        # RuntimeWarning that it is calculating the mean of an empty slice (with no non-Nans)
        # this is the desired behavior, but we want to avoid the warning here
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x)


def nan_max(x):
    with warnings.catch_warnings():
        # final mean should be mean of scores, ignoring NaN, hence nanmax
        # but if the group function values is NaN for ALL values, nanmean throws a
        # RuntimeWarning that it is calculating the mean of an empty slice (with no non-Nans)
        # this is the desired behavior, but we want to avoid the warning here
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(x)


class UpdateStream(InstanceOperator):
    update: dict

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance.update(self.update)
        return instance


class Metric(Artifact):
    main_score: str = AbstractField()
    # Override 'prediction_type' with the expected type of predictions
    # and references.  Example: "List[str]", "List[Dict]"", "string".
    # If left with default None, a warning will be displayed.
    # In future versions of unitxt, this will be an error.
    prediction_type: str = None

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
        if not isoftype(prediction, self.get_prediction_type()):
            raise ValueError(
                f"Each prediction is expected to be of type '{self.prediction_type}' in {self.get_metric_name()} metric. Received prediction of type {type(prediction)}: {prediction}"
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
            if not isoftype(ref, self.get_prediction_type()):
                raise ValueError(
                    f"Each reference is expected to be of type '{self.prediction_type}' in {self.get_metric_name()} metric. Received reference of type {type(ref)}: {ref}"
                )

    def get_prediction_type(self):
        if self.prediction_type is None:
            logger.warning(
                f"{self.get_metric_name()} metric does not set the 'prediction_type' parameter so input type checking is not performed. Set the prediction type to the expected prediction type (e.g. 'str', 'List[str]', or 'Any'). In future version of unitxt this will raise an exception."
            )
            self._parsed_prediction_type = Any
        try:
            if self._parsed_prediction_type is not None:
                return self._parsed_prediction_type

            self._parsed_prediction_type = parse_type_string(self.prediction_type)
        except ValueError:
            raise ValueError(
                f"Could convert prediction type '{self.prediction_type}' in {self.get_metric_name()} to known type.  To enable type checking for this prediction type, open unitxt issue with this message. Alternatively, set the metric's prediction_type to 'Any'"
            ) from None
        return self._parsed_prediction_type

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
    def disable_confidence_interval_calculation(self):
        pass


class MetricWithConfidenceInterval(Metric):
    # The number of resamples used to estimate the confidence intervals of this metric.
    # Use None to disable confidence interval computation.
    n_resamples: int = None
    confidence_level: float = 0.95
    ci_scores: List[str] = None

    @staticmethod
    def new_random_generator():
        # The np.random.default_rng expects a 32-bit int, while hash(..) can return a 64-bit integer.
        # So use '& MAX_32BIT' to get a 32-bit seed.
        _max_32bit = 2**32 - 1
        return np.random.default_rng(hash(get_seed()) & _max_32bit)

    def disable_confidence_interval_calculation(self):
        self.n_resamples = None

    def _can_compute_confidence_intervals(self, num_predictions):
        return (
            self.n_resamples is not None
            and self.n_resamples > 1
            and num_predictions > 1
        )

    @staticmethod
    def average_item_scores(instances: List[dict], score_name: str):
        """Calculate mean of a set of instance scores (given by score_name), omitting NaN values.

        Args:
            instances: list of dicts of each instance's instance scores.
            score_name: score field names to compute the mean for.
        """
        return nan_mean(
            [instance["score"]["instance"][score_name] for instance in instances]
        )

    @staticmethod
    def max_item_scores(instances: List[dict], score_name: str):
        """Calculate max of a set of instance scores (given by score_name), omitting NaN values.

        Args:
            instances: list of dicts of each instance's instance scores.
            score_name: score field names to compute the mean for.
        """
        return nan_max(
            [instance["score"]["instance"][score_name] for instance in instances]
        )

    @staticmethod
    def _all_instance_scores_equal(instances, score_name):
        instance_scores = [
            instance["score"]["instance"][score_name] for instance in instances
        ]
        non_nan_instance_scores = [
            score for score in instance_scores if score is not np.nan
        ]
        num_unique_scores = len(set(non_nan_instance_scores))
        return num_unique_scores == 1

    def score_based_confidence_interval(
        self,
        instances: List[dict],
        score_names: List[str],
        aggregation_func=None,
        ci_score_prefix="",
    ):
        """Compute confidence intervals based on existing scores, already computed on the input instances.

        Unlike GlobalMetric, this is simply a function of the instance scores (possibly taking into account task_data field),
         so they don't need to be recomputed after every bootstrap draw.

        Args:
            instances: The instances for which the confidence intervals are computed; should already have the relevant instance scores calculated.
            score_names: List of instance score field names to compute a confidence interval for.
            aggregation_func: A function with arguments instances, field_name; is applied on list of instances (which may include task_data
                field, as well as the prediction and references), and the field_name; default is simply to take the mean field_name from
                instances after resampling, if argument is None.
            ci_score_prefix: An optional string prefix to the score_name in the CI.  Useful in cases where the
                aggregation_func is something other than the mean

        Returns:
            Dict of confidence interval values
        """
        result = {}

        if not self._can_compute_confidence_intervals(num_predictions=len(instances)):
            return result

        ci_score_prefix = str(ci_score_prefix)
        if aggregation_func is None:
            # if aggregation_func is None, we simply take the mean of the resampled instance scores
            # otherwise, the aggregation_func needs to be applied AFTER resampling the instances;
            #   that is, re-form the groups, calculate the function, and take the mean of the group scores
            aggregation_func = self.average_item_scores
        for score_name in score_names:
            # If all computed instance level scores are the same, there is no point in computing
            # confidence intervals. So skip to the next score.
            if self._all_instance_scores_equal(instances, score_name):
                continue

            # need to redefine the statistic function within the loop because score_name is a loop variable
            def statistic(arr, axis, score_name=score_name):
                # arr is a 2d array where each row is a resampling, so we
                # iterate over the rows and compute the metric on each resampling
                scores = numpy.apply_along_axis(
                    lambda resampled_instances: aggregation_func(
                        resampled_instances, score_name
                    ),
                    axis=axis,
                    arr=arr,
                )
                return self.resample_from_non_nan(scores)

            # apply bootstrap only on the relevant field
            ci = bootstrap(
                (instances,),
                statistic=statistic,
                n_resamples=self.n_resamples,
                confidence_level=self.confidence_level,
                random_state=self.new_random_generator(),
            ).confidence_interval
            full_score_name = ci_score_prefix + score_name
            result[f"{full_score_name}_ci_low"] = ci.low
            result[f"{full_score_name}_ci_high"] = ci.high
            if score_name == self.main_score:
                result["score_ci_low"] = ci.low
                result["score_ci_high"] = ci.high
        return result

    def resample_from_non_nan(self, values):
        """Given an array values, will replace any NaN values with elements resampled with replacement from the non-NaN ones.

        here we deal with samples on which the metric could not be computed. These are
        edge cases - for example, when the sample contains only empty strings.
        CI is about the distribution around the statistic (e.g. mean), it doesn't deal with
        cases in which the metric is not computable. Therefore, we ignore these edge cases
        as part of the computation of CI.

        In theory there would be several ways to deal with this:
        1. skip the errors and return a shorter array => this fails because Scipy requires
        this callback (i.e. the statistic() callback) to return an array of the same size
        as the number of resamples
        2. Put np.nan for the errors => this fails because in such case the ci itself
        becomes np.nan. So one edge case can fail the whole CI computation.
        3. Replace the errors with a sampling from the successful cases => this is what is implemented.

        This resampling makes it so that, if possible, the bca confidence interval returned by bootstrap will not be NaN, since
        bootstrap does not ignore NaNs.  However, if there are 0 or 1 non-NaN values, or all non-NaN values are equal,
        the resulting distribution will be degenerate (only one unique value) so the CI will still be NaN since there is
        no variability.  In this case, the CI is essentially an interval of length 0 equaling the mean itself.
        """
        if values.size > 1:
            error_indices = numpy.isnan(values)
            n_errors = sum(error_indices)
            if 0 < n_errors < values.size:
                # replace NaN aggregate scores with random draws from non-NaN scores, so that confidence interval isn't NaN itself
                values[error_indices] = self.new_random_generator().choice(
                    values[~error_indices], n_errors, replace=True
                )
        return values

    def compute_global_confidence_intervals(
        self, references, predictions, task_data, score_name
    ):
        """Computed confidence intervals for a set of references and predictions."""
        random_gen = self.new_random_generator()

        def statistic(arr, axis):
            # arr is a 2d array where each row is a resampling, so we
            # iterate over the rows and compute the metric on each resampling
            def metric(sample_refs, sample_preds, sample_task_data):
                try:
                    return self._compute(
                        references=sample_refs,
                        predictions=sample_preds,
                        task_data=sample_task_data,
                    )["score"]
                except Exception as e:
                    # this happens in edge cases, for example, when the sampling creates a
                    # sample where all strings are empty and this fails bleu.
                    logger.info(f"Warning in {self.__class__.__name__}", e)
                    return np.nan

            # resample the instance scores, and then return the global score each time
            scores = numpy.apply_along_axis(
                lambda x: metric(
                    sample_refs=[references[i] for i in x],
                    sample_preds=[predictions[i] for i in x],
                    sample_task_data=[task_data[i] for i in x],
                ),
                axis=axis,
                arr=arr,
            )

            # in some resamplings of instances, the global score may be NaN since it cannot be computed;
            # in these cases, the bca confidence interval will be NaN because it does not ignore these values,
            # so we replace any NaN values with those resampled from the non-NaN ones.
            return self.resample_from_non_nan(scores)

        result = {}
        num_predictions = len(predictions)
        if self._can_compute_confidence_intervals(num_predictions=num_predictions):
            identifiers = list(range(num_predictions))

            with warnings.catch_warnings():
                # Avoid RuntimeWarning in bootstrap computation. This happens on small datasets where
                # the value of the computed global metric is the same on all resamplings.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ci = bootstrap(
                    (identifiers,),
                    statistic=statistic,
                    n_resamples=self.n_resamples,
                    confidence_level=self.confidence_level,
                    random_state=random_gen,
                ).confidence_interval
            result["score_ci_low"] = ci.low
            result["score_ci_high"] = ci.high
            result[f"{score_name}_ci_low"] = ci.low
            result[f"{score_name}_ci_high"] = ci.high
        return result


class GlobalMetric(StreamOperator, MetricWithConfidenceInterval):
    """A class for computing metrics that require joint calculations over all instances and are not just aggregation of scores of individuals instances.

    For example, macro_F1 requires
    calculation requires calculation of recall and precision per class, so all instances of the class
    need to be considered.  Accuracy, on the other hand, is just an average of the accuracy of all the instances.
    """

    n_resamples: int = OptionalField(
        default_factory=lambda: settings.num_resamples_for_global_metrics
    )

    # calculate scores for single instances
    process_single_instances = True

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        references = []
        predictions = []
        task_data = []
        global_score = {}

        instances = []

        for instance in stream:
            instance = self.verify_instance(instance)

            if "score" not in instance:
                instance["score"] = {"global": {}, "instance": {}}

            instance_references, instance_prediction = (
                instance["references"],
                instance["prediction"],
            )

            references.append(instance_references)
            predictions.append(instance_prediction)
            instances.append(instance)

            instance_task_data = (
                instance["task_data"] if "task_data" in instance else {}
            )
            task_data.append(instance_task_data)
            instance_score = None

            # for backward compatibility
            no_score_value = np.nan
            if self.process_single_instances:
                try:
                    instance_score = self._compute(
                        [instance_references],
                        [instance_prediction],
                        [instance_task_data],
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

            instance["score"]["instance"].update(
                self._add_score_prefixes_to_score_dict(instance_score)
            )
        self._validate_references_and_prediction(references, predictions)

        result = self._compute(references, predictions, task_data)

        global_score.update(self._add_score_prefixes_to_score_dict(result))
        score_name = global_score["score_name"]
        confidence_interval = self.compute_global_confidence_intervals(
            references, predictions, task_data, score_name
        )
        global_score.update(confidence_interval)

        for instance in instances:
            instance["score"]["global"].update(global_score)
            yield instance

    def _compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Any],
    ) -> dict:
        result = self.compute(references, predictions, task_data)
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result

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


class BulkInstanceMetric(StreamOperator, MetricWithConfidenceInterval):
    n_resamples: int = OptionalField(
        default_factory=lambda: settings.num_resamples_for_instance_metrics
    )
    main_score: str

    reduction_map: Dict[str, List[str]]

    implemented_reductions: List[str] = field(default_factory=lambda: ["mean"])

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        global_score = {}
        instances = []

        # consume the stream
        references, predictions = map(
            list,
            zip(
                *[
                    itemgetter("references", "prediction")(
                        self.verify_instance(instance)
                    )
                    for instance in stream
                ]
            ),
        )

        task_data = [
            instance["task_data"] if "task_data" in instance else {}
            for instance in stream
        ]
        self._validate_references_and_prediction(references, predictions)
        # compute the metric over all refs and preds
        instance_scores = self.compute(
            references=references,
            predictions=predictions,
            task_data=task_data,
        )

        # add the score and score_name fields
        for instance_score in instance_scores:
            instance_score["score"] = instance_score[self.main_score]
            instance_score["score_name"] = self.main_score

        for instance, score in zip(stream, instance_scores):
            if "score" not in instance:
                instance["score"] = {"global": {}, "instance": {}}

            instance["score"]["instance"].update(
                self._add_score_prefixes_to_score_dict(score)
            )
            instances.append(instance)

        for reduction, fields in self.reduction_map.items():
            assert (
                reduction in self.implemented_reductions
            ), f"Reduction {reduction} is not implemented, use one of {self.implemented_reductions}"

            if reduction == "mean":
                for field_name in fields:
                    field_name_with_prefix = self._add_score_prefix(field_name)
                    global_score[field_name_with_prefix] = mean(
                        [
                            instance["score"]["instance"][field_name_with_prefix]
                            for instance in instances
                        ]
                    )
                    if field_name == self.main_score:
                        global_score["score"] = global_score[field_name_with_prefix]
                        global_score["score_name"] = self.score_prefix + self.main_score

                ci_fields = (
                    list(set(self.ci_scores))
                    if self.ci_scores is not None
                    else [self.main_score]
                )
                ci_fields_with_prefix = [
                    self._add_score_prefix(ci_field) for ci_field in ci_fields
                ]
                confidence_interval = self.score_based_confidence_interval(
                    instances=instances, score_names=ci_fields_with_prefix
                )
                global_score.update(confidence_interval)

        for instance in instances:
            instance["score"]["global"].update(global_score)
            yield instance

    @abstractmethod
    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        pass


class InstanceMetric(StreamOperator, MetricWithConfidenceInterval):
    """Class for metrics for which a global score can be calculated by aggregating the instance scores (possibly with additional instance inputs).

    InstanceMetric currently allows two reductions:
    1. 'mean', which calculates the mean of instance scores,
    2. 'group_mean', which first applies an aggregation function specified in the reduction_map
        to instance scores grouped by the field grouping_field (which must not be None), and returns the mean
        of the group scores; if grouping_field is None, grouping is disabled.
        See _validate_group_mean_reduction for formatting instructions.
    """

    n_resamples: int = OptionalField(
        default_factory=lambda: settings.num_resamples_for_instance_metrics
    )

    # some group_mean aggregation functions (3rd element of "agg_func" list in the reduction)
    # only require a list of instance scores (e.g., mean, median, etc.).  Others aggregation functions
    # require an additional column (e.g., a subgroup identifier) by which the instance scores will be grouped
    # if subgroup_column is not None, a column by the specified name will be required in task_data
    subgroup_column = None
    implemented_reductions: List[str] = field(
        default_factory=lambda: ["mean", "group_mean", "max"]
    )

    reduction_map: Dict[str, List[str]] = AbstractField()

    reference_field: str = NonPositionalField(default="references")
    prediction_field: str = NonPositionalField(default="prediction")

    def _validate_group_mean_reduction(self, instances: List[dict]):
        """Ensure that group_mean reduction_map is properly formatted.

        Example: Apply the variance (np.var) to group Accuracy instance scores.  This class would be specified as follows:

        class GroupVarianceAccuracy(Accuracy):
            reduction_map = {'group_mean': {'agg_func': ['variance', np.var, True]}}

        reduction_map must be a dict with values containing
        - an 'agg_func' field with value being a 3-element list where
            - 1st element is a string name of the aggregation function (used in naming the CI report)
            - 2nd element is the callable aggregation function
            - 3rd element is a Boolean indicator of whether, during bootstrap CI calculation, the groups are to be sampled as single units.
                If True, the group scores are calculated and then resampled.  This treats the group units as the unit of
                interest for which the CI is being compared.
                If False, the instances are resampled individually, and the groups determined
                (meaning the groups may be of slightly different size or composition from the original
                depending on the resampling of the instances).
        - Optional: 'score_fields' key with list value containing the string names of fields to apply the aggregation to
            - If not present, the parent class main_score is used.

        The aggregation function (2nd element of agg_func) can be one of two types:
        1. simple: calculate a summary statistic from a single group of values (e.g. mean, median, etc.).
            This is best suited for cases where the instances are independent of each other, other than belonging to the same group
        2. comparison: requires subgroup_column to be specified.  This function conducts
            a comparison between scores for differing values of subgroup_column (e.g., 'original' vs 'paraphrase').
            An example is where the original instance is a question, and the others are various paraphrases
            or perturbations of this question.  Here, the function would return, say, a comparison of the instance accuracies
            rather than, say, the average instance accuracy.
            In these cases, we recommend setting the 3rd parameter to be True so that the groups are resampled together.

        Example:
            class GroupVsBaselineDiffAccuracy(Accuracy):
                subgroup_column = 'variant_type'
                reduction_map = {'group_mean': {'agg_func': ['accuracy_diff', accuracy_diff, True],}}

            # where the function is defined as
            def accuracy_diff(subgroup_scores_dict, expected_subgroup_types=['original', 'paraphrase']):
                validate_subgroup_types(subgroup_scores_dict, expected_subgroup_types)
                from statistics import mean
                return mean(subgroup_scores_dict['paraphrase']) - mean(subgroup_scores_dict['original'])
            The input dataset should look like:

            'group_id'  'question'                                   'variant_type'
            1           'How do you fix a car engine?'               'original'
            1           'What is the best way to fix an engine?'     'paraphrase'
            1           'How do you repair a car engine?'            'paraphrase'
            1           'How do I repair my engine?'                 'paraphrase'
            2           'Why are ants eating my food?'               'original'
        """
        # instances need to all have task_data field with field group_id
        assert all(
            "task_data" in instance for instance in instances
        ), "each instance must have an task_data field"
        assert all(
            isinstance(instance["task_data"], dict) for instance in instances
        ), "each instance must have an task_data field that is a dict"
        assert all(
            "group_id" in instance["task_data"] for instance in instances
        ), "each instance task_data dict must have a key group_id"

        # validate the reduction_map
        assert (
            "group_mean" in self.reduction_map
        ), "reduction_map must have a 'group_mean' key"
        fields = self.reduction_map["group_mean"]
        # for group_mean, expects a dict
        assert isinstance(fields, dict)
        assert (
            "agg_func" in fields
        ), "fields should have a key 'agg_func' whose value is a 3-element list of a function name, function definition, and a boolean indicator"
        assert isinstance(
            fields["agg_func"], list
        ), "fields['agg_func'] should be a list"
        assert (
            len(fields["agg_func"]) == 3
        ), "fields['agg_func'] should be a 3-element list"
        assert isinstance(
            fields["agg_func"][0], str
        ), "first item in fields['agg_func'] should be a string name of a function"
        assert callable(
            fields["agg_func"][1]
        ), "second item in fields['agg_func'] should be a callable function"
        assert isinstance(
            fields["agg_func"][2], bool
        ), "third item in fields['agg_func'] should be a boolean value"
        if "score_fields" in fields:
            assert isinstance(fields["score_fields"], list)

        # for aggregation functions that use the subgroup_column (expect a dict of lists), check that
        # this field exists
        if self.subgroup_column is not None:
            assert all(
                self.subgroup_column in instance["task_data"] for instance in instances
            ), f"each instance task_data dict must have a key {self.subgroup_column}"

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        instances = self.compute_instance_scores(stream)
        global_score = {}
        for reduction_type, reduction_params in self.reduction_map.items():
            assert (
                reduction_type in self.implemented_reductions
            ), f"Reduction {reduction_type} is not implemented, use one of {self.implemented_reductions}"

            field_name_full_prefix = ""
            # used for passing to the bootstrapping, depends on whether the groups are fixed or not
            aggregation_function = None
            if reduction_type == "mean":
                aggregation_function = self.average_item_scores
                reduction_fields = list(set(reduction_params))
                # no group reduction, so resample instances individually
                scores_to_resample = instances
            elif reduction_type == "max":
                aggregation_function = self.max_item_scores
                reduction_fields = list(set(reduction_params))
                # no group reduction, so resample instances individually
                scores_to_resample = instances
            elif reduction_type == "group_mean":
                aggregation_function = self.average_item_scores
                self._validate_group_mean_reduction(instances=instances)
                reduction_fields = (
                    [self.main_score]
                    if "score_fields" not in reduction_params
                    else list(set(reduction_params["score_fields"]))
                )
                aggregation_function_name = str(reduction_params["agg_func"][0])
                field_name_full_prefix = "group_" + aggregation_function_name + "_"
                do_resample_as_group = reduction_params["agg_func"][2]
                if do_resample_as_group:
                    # append fixed_ to name because resamples the groups as fixed units
                    field_name_full_prefix = "fixed_" + field_name_full_prefix
                (
                    scores_to_resample,
                    aggregation_function,
                ) = self._set_up_group_mean_aggregation(
                    instances, reduction_params, reduction_fields
                )
            else:
                raise ValueError(
                    f"Reduction {reduction_type} is not supported, please specify a valid reduction method in reduction_map {self.reduction_map}."
                )

            # calculate global scores for each reduction field
            for field_name in reduction_fields:
                field_name_full = (
                    field_name_full_prefix + self.score_prefix + field_name
                )
                # if group resampling (3rd element of agg_func parameter) is True, then
                #   1. scores_to_resample are the group scores, and
                #   2. aggregation_function is to take the raw mean
                # if no group resampling (3rd element of agg_func parameter) is False, then
                #   1. scores_to_resample are the original instance scores, and
                #   2. aggregation_function is to apply the group aggregation from the instance scores
                # either way, the application of aggregation_function to scores_to_resample yields the global score
                global_score[field_name_full] = aggregation_function(
                    scores_to_resample, self.score_prefix + field_name
                )
                if field_name == self.main_score:
                    global_score["score"] = global_score[field_name_full]
                    global_score["score_name"] = field_name_full

            # need to specify which fields should have CIs calculated for them through ci_scores
            # (will not automatically calculate CIs for fields in reduction map)
            if self.ci_scores is not None:
                confidence_interval = self.score_based_confidence_interval(
                    instances=scores_to_resample,
                    score_names=[
                        self.score_prefix + ci_score for ci_score in set(self.ci_scores)
                    ],
                    ci_score_prefix=field_name_full_prefix,
                    aggregation_func=aggregation_function,
                )
                global_score.update(confidence_interval)

        for instance in instances:
            instance["score"]["global"].update(global_score)
        yield from instances

    def compute_instance_scores(
        self, stream: Stream, stream_name: Optional[str] = None
    ):
        instances = []

        for instance in stream:
            instance = self.verify_instance(instance)

            task_data = instance["task_data"] if "task_data" in instance else {}

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

            self._validate_prediction(pred)
            self._validate_reference(refs)

            instance_score = self.compute(
                references=refs, prediction=pred, task_data=task_data
            )

            instance_score["score"] = instance_score[self.main_score]
            instance_score["score_name"] = self.main_score
            if "score" not in instance:
                instance["score"] = {"global": {}, "instance": {}}

            instance["score"]["instance"].update(
                self._add_score_prefixes_to_score_dict(instance_score)
            )

            instances.append(instance)

        return instances

    def get_group_scores(
        self, instances: List[dict], score_names: List[str], group_aggregation_func
    ):
        """Group scores by the group_id and subgroup_type fields of each instance, and compute group_aggregation_func by group.

        Args:
            instances: List of observation instances with instance-level scores (fields) computed.
            score_names: List of instance score names in each instance to apply the aggregation function.
            group_aggregation_func: Callable aggregation function accepting a list of numeric scores;
                or, if self.subgroup_column is not None, a dict of subgroup types scores by subgroup_column value.
                callable function returns a single score for the group

        Returns:
            List of dicts, each corresponding to a group of instances (defined by 'group_id'),
                with an aggregate group score for each score_name
        """
        from collections import defaultdict

        # three-level defaultdict:
        # first is the grouping, second is the field name, the third is the subgroup_type (by default 'default')
        group_to_instance_scores = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        # check if function has fields for subgroup_column
        uses_subgroups = self.subgroup_column is not None
        default_subgroup_name = "default"
        # loop through the instances and group the scores
        for instance in instances:
            task_data = instance["task_data"]
            group_key = task_data["group_id"]
            # for functions that do comparisons between subgroup_column groups
            # if function doesn't use subgroup_column, or none is present, set "default" as default value, and pass all scores
            subgroup_type = (
                task_data[self.subgroup_column]
                if uses_subgroups
                else default_subgroup_name
            )
            for score_name in score_names:
                group_to_instance_scores[group_key][score_name][subgroup_type].append(
                    instance["score"]["instance"][score_name]
                )

        # if group_aggregation_func expects a subgroup-types score dict, pass it; otherwise pass the default type list of scores
        return [
            {
                "score": {
                    "instance": {
                        score_name: group_aggregation_func(
                            score_dict
                            if uses_subgroups
                            else score_dict[default_subgroup_name]
                        )
                        for score_name, score_dict in group_to_instance_scores[
                            group_name
                        ].items()
                    }
                }
            }
            for group_name in sorted(
                group_to_instance_scores.keys()
            )  # sorted for consistency
        ]

    def _set_up_group_mean_aggregation(
        self, instances, reduction_params, reduction_fields
    ):
        group_aggregation_func = reduction_params["agg_func"][1]
        # if treat groups as units
        do_resample_as_group = reduction_params["agg_func"][2]
        if do_resample_as_group:
            # pass the group aggregate---not instance---scores to resample as usual
            aggregation_function = self.average_item_scores
            scores_to_resample = self.get_group_scores(
                instances, reduction_fields, group_aggregation_func
            )
        else:
            # pass the instance scores to resample, and calculate the group aggregation on the resamplings
            scores_to_resample = instances

            def aggregation_function(
                instances,
                field_name,
                group_aggregation_func=group_aggregation_func,
            ):
                group_scores = self.get_group_scores(
                    instances, [field_name], group_aggregation_func
                )
                return nan_mean(
                    [group["score"]["instance"][field_name] for group in group_scores]
                )

        return scores_to_resample, aggregation_function

    @abstractmethod
    def compute(self, references: List[Any], prediction: Any, task_data: Dict) -> dict:
        pass


class Accuracy(InstanceMetric):
    reduction_map = {"mean": ["accuracy"]}
    main_score = "accuracy"
    ci_scores = ["accuracy"]

    prediction_type = "Any"  # string representation is compared

    def compute(
        self, references: List[Any], prediction: Any, task_data: List[Dict]
    ) -> dict:
        result = {
            self.main_score: float(
                str(prediction) in [str(reference) for reference in references]
            )
        }
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result


class JaccardIndex(InstanceMetric):
    reduction_map = {"mean": ["jaccard_index"]}
    main_score = "jaccard_index"
    ci_scores = ["jaccard_index"]

    prediction_type = "Any"  # string representation is compared

    def compute(
        self, references: List[Any], prediction: Any, task_data: List[Dict]
    ) -> dict:
        if not isinstance(prediction, set):
            prediction = set(prediction)
        references = [set(reference) for reference in references]

        result = {
            self.main_score: max(
                [
                    float(
                        (len(reference.intersection(prediction)))
                        / (
                            len(reference)
                            + len(prediction)
                            - len(reference.intersection(prediction))
                        )
                    )
                    for reference in references
                ]
            )
        }
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result


class MaxAccuracy(Accuracy):
    """Calculate the maximal accuracy over all instances as the global score."""

    reduction_map = {"max": ["accuracy"]}


class UnsortedListExactMatch(InstanceMetric):
    reduction_map = {"mean": ["unsorted_list_exact_match"]}
    main_score = "unsorted_list_exact_match"
    ci_scores = ["unsorted_list_exact_match"]

    def compute(
        self, references: List[Any], prediction: Any, task_data: List[Dict]
    ) -> dict:
        result = {self.main_score: float(sorted(prediction) == sorted(references[0]))}
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result


class StringContainment(InstanceMetric):
    reduction_map = {"mean": ["string_containment"]}
    main_score = "string_containment"
    ci_scores = ["string_containment"]

    prediction_type = "Any"  # string representation is compared
    single_reference_per_prediction = False  # multiple references allowed

    def compute(
        self, references: List[Any], prediction: Any, task_data: List[Dict]
    ) -> dict:
        result = {
            self.main_score: float(
                any(str(reference) in str(prediction) for reference in references)
            )
        }
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result


class MetricPipeline(MultiStreamOperator, Metric):
    main_score: str = None
    preprocess_steps: Optional[List[StreamingOperator]] = field(default_factory=list)
    postpreprocess_steps: Optional[List[StreamingOperator]] = field(
        default_factory=list
    )
    metric: Metric = None

    def disable_confidence_interval_calculation(self):
        self.metric.disable_confidence_interval_calculation()

    def verify(self):
        assert (
            self.metric is not None
        ), f"'metric' is not set in {self.get_metric_name()}"
        assert (
            self.main_score is not None
        ), f"'main_score' is not set in {self.get_metric_name()}"
        assert isinstance(
            self.metric, Metric
        ), f"'metric' is not set to a Metric class in {self.get_metric_name()} (type{self.metric})"

    def prepare(self):
        super().prepare()
        self.prepare_score = Copy(
            field_to_field=[
                [
                    f"score/instance/{self.metric._add_score_prefix(self.main_score)}",
                    "score/instance/score",
                ],
                [
                    f"score/global/{self.metric._add_score_prefix(self.main_score)}",
                    "score/global/score",
                ],
            ],
        )

    def process(self, multi_stream: MultiStream) -> MultiStream:
        for step in self.preprocess_steps:
            multi_stream = step(multi_stream)
        multi_stream = self.metric(multi_stream)
        for step in self.postpreprocess_steps:
            multi_stream = step(multi_stream)
        return self.prepare_score(multi_stream)


class HuggingfaceMetric(GlobalMetric):
    hf_metric_name: str = None
    main_score: str = None  # The main score returned from the metric
    hf_main_score: str = (
        None  # USed if HF returns uses a different score name for the main metric
    )

    scale: float = 1.0  # optional scaling of main results
    scaled_fields: list = None
    # This are fixed arguments  passed to compute method
    hf_compute_args: Dict[str, Any] = OptionalField(default_factory=dict)
    # These are additional input fields passed to HF compute method (a list with one value per instance)
    hf_additional_input_fields: List = OptionalField(default_factory=list)
    # These are additional input fields that are passed as one value
    hf_additional_input_fields_pass_one_value: List = OptionalField(
        default_factory=list
    )

    experiment_id: str = OptionalField(default_factory=lambda: str(uuid.uuid4()))

    def verify(self):
        assert (
            self.hf_additional_input_fields is None
            or isoftype(self.hf_additional_input_fields, List[str])
        ), f"Argument hf_additional_input_fields should be either None or List[str]. It is now: {self.hf_additional_input_fields}."
        assert (
            self.hf_additional_input_fields_pass_one_value is None
            or isoftype(self.hf_additional_input_fields_pass_one_value, List[str])
        ), f"Argument hf_additional_input_fields_pass_one_value should be either None or List[str]. It is now: {self.hf_additional_input_fields_pass_one_value}."

        return super().verify()

    def prepare(self):
        super().prepare()
        self.metric = evaluate.load(
            self.hf_metric_name, experiment_id=self.experiment_id
        )

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> dict:
        passed_task_data = {}
        for additional_input_field in self.hf_additional_input_fields:
            assert (
                additional_input_field in task_data[0]
            ), f"'{additional_input_field}' field required by {__class__.__name__} is not in passed in task_data: {task_data[0]}"
            passed_task_data[additional_input_field] = [
                additional_input[additional_input_field]
                for additional_input in task_data
            ]
        for additional_input_field in self.hf_additional_input_fields_pass_one_value:
            assert (
                additional_input_field in task_data[0]
            ), f"'{additional_input_field}' field required by {__class__.__name__} is not in passed in task_data: {task_data[0]}"

            values = {
                additional_input[additional_input_field]
                for additional_input in task_data
            }
            assert (
                len(values) == 1
            ), f"Values of '{additional_input_field}' field required by {__class__.__name__}  should all be the same, but have multiple values {values}"

            passed_task_data[additional_input_field] = next(iter(values))

        # add check that all required fields in self.metrics are in passed_task_data
        result = self.metric.compute(
            predictions=predictions,
            references=references,
            **passed_task_data,
            **self.hf_compute_args,
        )
        if self.hf_main_score:
            result[self.main_score] = result[self.hf_main_score]
            del result[self.hf_main_score]
        if self.scale != 1.0:
            assert (
                self.scaled_fields is not None
            ), f"Scaling factor was set to {self.scale}, but no fields specified"
            for key in self.scaled_fields:
                assert (
                    key in result
                ), f"Trying to scale field '{key}' which is not in results of metrics: {result}"
                if isinstance(result[key], list):
                    assert all(
                        isinstance(v, float) for v in result[key]
                    ), "Not all scaled field '{key}' values are floats: {result[key]}"
                    result[key] = [v / self.scale for v in result[key]]
                else:
                    assert isinstance(
                        result[key], float
                    ), "Scaled field '{key}' is not float: {result[key]}"
                    result[key] /= self.scale
        return result


class HuggingfaceBulkMetric(BulkInstanceMetric):
    hf_metric_name: str

    hf_metric_fields: List[str]
    hf_compute_args: dict = {}
    hf_additional_input_fields: List = OptionalField(default_factory=list)

    def prepare(self):
        super().prepare()
        self.metric = evaluate.load(
            self.hf_metric_name, experiment_id=str(uuid.uuid4())
        )

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Any],
    ) -> List[Dict[str, Any]]:
        passed_task_data = {}
        for additional_input_field in self.hf_additional_input_fields:
            assert (
                additional_input_field in task_data[0]
            ), f"'{additional_input_field}' field required by {__class__.__name__} is not in passed in task_data: {task_data[0]}"
            passed_task_data[additional_input_field] = [
                additional_input[additional_input_field]
                for additional_input in task_data
            ]
        # add check that all required fields in self.metrics are in passed_task_data

        scores = self.metric.compute(
            predictions=predictions,
            references=references,
            **passed_task_data,
            **self.hf_compute_args,
        )

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

    prediction_type = "str"
    single_reference_per_prediction = True

    def prepare(self):
        super().prepare()
        self._metric = evaluate.load(self.metric, experiment_id=str(uuid.uuid4()))

    def get_str_id(self, str):
        if str not in self.str_to_id:
            id = len(self.str_to_id)
            self.str_to_id[str] = id
            self.id_to_str[id] = str
        return self.str_to_id[str]

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict],
    ) -> dict:
        self.str_to_id = {}
        self.id_to_str = {}
        formatted_references = [
            self.get_str_id(reference[0]) for reference in references
        ]
        self.str_to_id.keys()
        formatted_predictions = [
            self.get_str_id(prediction) for prediction in predictions
        ]
        labels = list(set(formatted_references))

        result = self._metric.compute(
            predictions=formatted_predictions,
            references=formatted_references,
            labels=labels,
            average=self.average,
        )
        if isinstance(result[self.metric], numpy.ndarray):
            final_result = {self.main_score: mean(result[self.metric])}
            for i, label in enumerate(labels):
                final_result[f"{self.metric}_" + self.id_to_str[label]] = result[
                    self.metric
                ][i]
        else:
            final_result = {self.main_score: result[self.metric]}
        return final_result


class F1Micro(F1):
    main_score = "f1_micro"
    average = "micro"


class F1Binary(GlobalMetric):
    """Calculate f1 for a binary task, using 0.5 as the threshold in the case of float predictions."""

    process_single_instances = False
    main_score = "f1_binary"
    average = None
    threshold = 0.5
    prediction_type = "Union[float, int]"
    _metric = None
    metric = "f1"
    single_reference_per_prediction = True
    _requirements_list: List[str] = ["sklearn"]

    def prepare(self):
        super().prepare()
        from sklearn import metrics

        self._metric = metrics.precision_recall_fscore_support

    def _validate_reference(self, reference):
        super()._validate_reference(reference)
        assert reference[0] in [
            0,
            1,
        ], f"all references of {self.main_score} must by 0 or 1"

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict],
    ) -> dict:
        flattened_int_references = [int(r[0]) for r in references]
        int_predictions = [int(p > self.threshold) for p in predictions]
        precision, recall, f1, _ = self._metric(
            y_true=flattened_int_references,
            y_pred=int_predictions,
            labels=[0, 1],
            average=self.average,
        )
        if self.average is None:
            return {
                "f1_binary": f1[1],
                "f1_binary_neg": f1[0],
                "recall_binary": recall[1],
                "recall_binary_neg": recall[0],
                "precision_binary": precision[1],
                "precision_binary_neg": precision[0],
            }
        return {"f1_binary": f1, "recall_binary": recall, "precision_binary": precision}


class F1BinaryPosOnly(F1Binary):
    average = "binary"
    main_score = "f1_binary"


class RecallBinary(F1Binary):
    main_score = "recall_binary"
    metric = "recall"


class PrecisionBinary(F1Binary):
    main_score = "precision_binary"
    metric = "precision"


class F1Macro(F1):
    main_score = "f1_macro"


class F1Weighted(F1):
    main_score = "f1_weighted"
    average = "weighted"


class F1MultiLabel(GlobalMetric):
    _metric = None
    main_score = "f1_macro"
    average = None  # Report per class then aggregate by mean
    metric = "f1"

    prediction_type = "List[str]"
    single_reference_per_prediction = True

    def prepare(self):
        super().prepare()
        self._metric = evaluate.load(
            self.metric, "multilabel", experiment_id=str(uuid.uuid4())
        )

    def add_str_to_id(self, str):
        if str not in self.str_to_id:
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

    def compute(
        self,
        references: List[List[str]],
        predictions: List[List[str]],
        task_data: List[Dict],
    ) -> dict:
        self.str_to_id = {}
        self.id_to_str = {}

        references = [reference[0] for reference in references]

        labels = list({label for reference in references for label in reference})

        # if no classes are left then F1 is not defined
        if len(labels) == 0:
            return {self.main_score: float("nan")}

        for label in labels:
            self.add_str_to_id(label)
        formatted_references = [
            self.get_one_hot_vector(reference) for reference in references
        ]
        formatted_predictions = [
            self.get_one_hot_vector(prediction) for prediction in predictions
        ]

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
        if isinstance(result[self.metric], numpy.ndarray):
            assert (
                len(result[self.metric]) == len(labels)
            ), f"F1 result ({result[self.metric]}) has more entries than labels ({labels})"
            final_result = {self.main_score: mean(result[self.metric])}
            for i, label in enumerate(labels):
                final_result[self.metric + "_" + label] = result[self.metric][i]
        else:
            final_result = {self.main_score: result[self.metric]}
        return final_result


class PrecisionMacroMultiLabel(F1MultiLabel):
    main_score = "precision_macro"
    metric = "precision"
    average = "macro"


class PrecisionMicroMultiLabel(F1MultiLabel):
    main_score = "precision_micro"
    metric = "precision"
    average = "micro"


class RecallMacroMultiLabel(F1MultiLabel):
    main_score = "recall_macro"
    metric = "recall"
    average = "macro"


class RecallMicroMultiLabel(F1MultiLabel):
    main_score = "recall_micro"
    metric = "recall"
    average = "micro"


class F1MicroMultiLabel(F1MultiLabel):
    main_score = "f1_micro"
    average = "micro"


class F1MacroMultiLabel(F1MultiLabel):
    main_score = "f1_macro"
    average = None


class Rouge(HuggingfaceMetric):
    hf_metric_name = "rouge"
    main_score = "rougeL"
    scale = 1.0

    prediction_type = "str"
    single_reference_per_prediction = False  # multiple references allowed

    use_aggregator: bool = True
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    sent_split_newline: bool = True

    _requirements_list: List[str] = ["nltk", "rouge_score"]

    def prepare(self):
        super().prepare()

        self.hf_compute_args.update(
            {"use_aggregator": self.use_aggregator, "rouge_types": self.rouge_types}
        )

        import nltk

        nltk.download("punkt")
        self.sent_tokenize = nltk.sent_tokenize

    def compute(self, references, predictions, task_data: List[Dict]):
        if self.sent_split_newline:
            predictions = [
                "\n".join(self.sent_tokenize(prediction.strip()))
                for prediction in predictions
            ]
            references = [
                ["\n".join(self.sent_tokenize(r.strip())) for r in reference]
                for reference in references
            ]
        return super().compute(references, predictions, task_data)


# Computes char edit distance, ignoring whitespace
class CharEditDistance(InstanceMetric):
    main_score = "char_edit_distance"
    reduction_map = {"mean": [main_score]}
    ci_scores = [main_score]
    prediction_type = "str"
    single_reference_per_prediction = True

    accuracy_metric = False

    _requirements_list: List[str] = ["editdistance"]

    def prepare(self):
        super().prepare()
        import editdistance

        self.eval = editdistance.eval

    def compute(self, references, prediction: str, task_data: List[Dict]) -> dict:
        formatted_prediction = "".join(prediction.split())
        formatted_reference = "".join(references[0].split())
        max_length = max(len(formatted_reference), len(formatted_prediction))
        if max_length == 0:
            return {self.main_score: 0.0}
        edit_dist = self.eval(formatted_reference, formatted_prediction)
        if self.accuracy_metric:
            score = 1 - edit_dist / max_length
        else:
            score = edit_dist
        return {self.main_score: score}


class CharEditDistanceAccuracy(CharEditDistance):
    main_score = "char_edit_dist_accuracy"
    reduction_map = {"mean": [main_score]}
    ci_scores = [main_score]

    accuracy_metric = True


class Wer(HuggingfaceMetric):
    hf_metric_name = "wer"
    main_score = "wer"
    prediction_type = "str"
    single_reference_per_prediction = True

    _requirements_list: List[str] = ["jiwer"]

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict],
    ) -> dict:
        formatted_references = [reference[0] for reference in references]
        result = self.metric.compute(
            predictions=predictions, references=formatted_references
        )
        return {self.main_score: result}


class Spearmanr(HuggingfaceMetric):
    hf_metric_name = "spearmanr"
    main_score = "spearmanr"
    process_single_instances = False
    prediction_type = "float"

    # Spearmanr references are not list
    def _validate_reference(self, reference):
        if not isoftype(reference, self.get_prediction_type()):
            raise ValueError(
                f"Each reference is expected to be of type '{self.prediction_type}' in {self.get_metric_name()} metric. Received prediction of type {type(reference)}: {reference}"
            )


class KendallTauMetric(GlobalMetric):
    main_score = "kendalltau_b"
    variant = "b"
    process_single_instances = False
    prediction_type = "float"

    _requirements_list: List[str] = ["scipy"]

    def prepare(self):
        from scipy.stats import kendalltau

        self.kendalltau = kendalltau

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict],
    ) -> dict:
        if isinstance(references[0], list):
            references = [reference[0] for reference in references]

        kendall_results = self.kendalltau(references, predictions, variant=self.variant)
        corr = kendall_results.correlation
        return {
            self.main_score: corr,
            f"{self.main_score}_p_val": kendall_results.pvalue,
        }


class MatthewsCorrelation(HuggingfaceMetric):
    hf_metric_name = "matthews_correlation"
    main_score = "matthews_correlation"
    str_to_id: dict = InternalField(default_factory=dict)

    single_reference_per_prediction = True
    prediction_type = "str"

    def get_str_id(self, str):
        if str not in self.str_to_id:
            id = len(self.str_to_id)
            self.str_to_id[str] = id
        return self.str_to_id[str]

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict],
    ) -> dict:
        formatted_references = [
            self.get_str_id(reference[0]) for reference in references
        ]
        formatted_predictions = [
            self.get_str_id(prediction) for prediction in predictions
        ]
        return self.metric.compute(
            predictions=formatted_predictions, references=formatted_references
        )


class RocAuc(GlobalMetric):
    main_score = "roc_auc"
    process_single_instances = False
    _requirements_list: List[str] = ["sklearn"]
    single_reference_per_prediction = True
    prediction_type = "float"

    def prepare(self):
        from sklearn import metrics

        self.roc_curve = metrics.roc_curve
        self.auc = metrics.auc

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict],
    ) -> dict:
        if isinstance(references[0], list):
            references = [reference[0] for reference in references]

        false_positive_rates, true_positive_rates, _ = self.roc_curve(
            y_true=references, y_score=predictions
        )
        roc_auc = self.auc(false_positive_rates, true_positive_rates)
        return {self.main_score: roc_auc}


class CustomF1(GlobalMetric):
    main_score = "f1_micro"
    prediction_type = "Any"
    single_reference_per_prediction = True
    groups = None
    zero_division: float = 0.0
    report_per_group_scores: bool = True

    @abstractmethod
    def get_element_group(self, element, additional_input):
        pass

    @abstractmethod
    def get_element_representation(self, element, additional_input):
        pass

    def should_ignore_element(self, element, additional_input):
        return False

    def group_elements(self, elements_list, additional_input):
        if not isinstance(elements_list, list):
            elements_list = [elements_list]
        return {
            k: Counter(
                [
                    self.get_element_representation(value, additional_input)
                    for value in elements_list
                    if self.get_element_group(value, additional_input) == k
                ]
            )
            for k in {
                self.get_element_group(e, additional_input)
                for e in elements_list
                if not self.should_ignore_element(e, additional_input)
            }
        }

    def calculate_groups_ratio(self, actual_group, total_group):
        return sum(
            [min(actual_group[k], total_group[k]) for k in actual_group.keys()]
        ), sum(actual_group.values())

    def precision(self, pn, pd, rn, rd):
        return self.zero_division if pn == 0 and pd == 0 else pn / pd

    def recall(self, pn, pd, rn, rd):
        return self.zero_division if rn == 0 and rd == 0 else rn / rd

    def f1(self, pn, pd, rn, rd):
        precision = self.precision(pn, pd, rn, rd)
        recall = self.recall(pn, pd, rn, rd)
        try:
            return 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            return self.zero_division

    def get_groups(self, elements, task_data):
        groups = set()
        for sublist, additional_input in zip(elements, task_data):
            if not isinstance(sublist, list):
                sublist = [sublist]
            for e in sublist:
                if self.should_ignore_element(e, additional_input):
                    continue
                groups.add(self.get_element_group(e, additional_input))
        return groups

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> dict:
        references = [element[0] for element in references]

        if self.groups is None:
            groups = self.get_groups(references, task_data)
        else:
            groups = self.groups
        groups_statistics = {}
        for references_batch, predictions_batch, additional_input in zip(
            references, predictions, task_data
        ):
            grouped_references = self.group_elements(references_batch, additional_input)
            grouped_predictions = self.group_elements(
                predictions_batch, additional_input
            )
            all_groups = set(grouped_references.keys()).union(
                grouped_predictions.keys()
            )
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

        num_of_unknown_class_predictions = 0
        pn_total = pd_total = rn_total = rd_total = 0
        f1_result = {}
        recall_result = {}
        precision_result = {}
        for group in groups_statistics.keys():
            pn, pd, rn, rd = (
                groups_statistics[group]["precision_numerator"],
                groups_statistics[group]["precision_denominator"],
                groups_statistics[group]["recall_numerator"],
                groups_statistics[group]["recall_denominator"],
            )
            pn_total, pd_total, rn_total, rd_total = (
                pn_total + pn,
                pd_total + pd,
                rn_total + rn,
                rd_total + rd,
            )
            if group in groups:
                f1_result[f"f1_{group}"] = self.f1(pn, pd, rn, rd)
                recall_result[f"recall_{group}"] = self.recall(pn, pd, rn, rd)
                precision_result[f"precision_{group}"] = self.precision(pn, pd, rn, rd)
            else:
                num_of_unknown_class_predictions += pd

        result = f1_result
        self.add_macro_scores(f1_result, recall_result, precision_result, result)
        self.add_in_class_support_scores(
            num_of_unknown_class_predictions, pd_total, result
        )
        self.add_micro_scores(rd_total, rn_total, pd_total, pn_total, result)
        if not self.report_per_group_scores:
            for group in groups:
                del result[f"f1_{group}"]
        return result

    def add_micro_scores(self, rd_total, rn_total, pd_total, pn_total, result):
        result["f1_micro"] = self.f1(pn_total, pd_total, rn_total, rd_total)
        result["recall_micro"] = self.recall(pn_total, pd_total, rn_total, rd_total)
        result["precision_micro"] = self.precision(
            pn_total, pd_total, rn_total, rd_total
        )

    def add_in_class_support_scores(
        self, num_of_unknown_class_predictions, pd_total, result
    ):
        amount_of_predictions = pd_total
        if amount_of_predictions == 0:
            result["in_classes_support"] = 1.0
        else:
            result["in_classes_support"] = (
                1.0 - num_of_unknown_class_predictions / amount_of_predictions
            )

    def add_macro_scores(self, f1_result, recall_result, precision_result, result):
        try:
            result["f1_macro"] = sum(f1_result.values()) / len(result.keys())
            result["recall_macro"] = sum(recall_result.values()) / len(
                recall_result.keys()
            )
            result["precision_macro"] = sum(precision_result.values()) / len(
                precision_result.keys()
            )
        except ZeroDivisionError:
            result["f1_macro"] = self.zero_division
            result["recall_macro"] = self.zero_division
            result["precision_macro"] = self.zero_division


class NER(CustomF1):
    prediction_type = "List[Tuple[str,str]]"

    def get_element_group(self, element, additional_input):
        return element[1]

    def get_element_representation(self, element, additional_input):
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
    ci_scores = ["f1", "precision", "recall"]
    single_reference_per_prediction = False
    prediction_type = "str"

    def compute(
        self, references: List[Any], prediction: Any, task_data: List[Dict]
    ) -> dict:
        results = [
            self._compute_single_ref(str(reference), str(prediction))
            for reference in references
        ]
        return {
            measure: max(r[i] for r in results)
            for i, measure in enumerate(["precision", "recall", "f1"])
        }

    def _compute_single_ref(
        self, reference: Any, prediction: Any
    ) -> Tuple[float, float, float]:
        prediction_tokens = normalize_answer(str(prediction)).split()
        reference_tokens = normalize_answer(str(reference)).split()
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
    hf_metric_name = "bertscore"
    main_score = "f1"
    reduction_map = {"mean": ["f1", "precision", "recall"]}
    hf_metric_fields = ["f1", "precision", "recall"]
    ci_scores = ["f1", "precision", "recall"]
    model_name: str
    model_layer: int = None

    prediction_type = "str"

    _requirements_list: List[str] = ["bert_score"]

    def prepare(self):
        super().prepare()
        self.hf_compute_args = {"model_type": self.model_name, "batch_size": 32}
        if self.model_layer:
            self.hf_compute_args["num_layers"] = self.model_layer


class SentenceBert(BulkInstanceMetric):
    reduction_map = {"mean": ["score"]}
    main_score = "score"
    batch_size: int = 32

    model_name: str

    _requirements_list: List[str] = ["sentence_transformers", "torch", "transformers"]

    def prepare(self):
        super().prepare()
        import torch
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util as sbert_util

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.util = sbert_util

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
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
        preds_emb = self.model.encode(predictions, device=self.device)
        refs_emb = self.model.encode(
            [ref for ref_group in references for ref in ref_group], device=self.device
        )

        # for each candidate, pick the reference with the highest score
        for pred_emb, ref_group_bounds in zip(preds_emb, ref_group_boundaries):
            refs_group_emb = refs_emb[ref_group_bounds[0] : ref_group_bounds[1]]
            scores.append(self.util.cos_sim(pred_emb, refs_group_emb).max().item())

        return [{"score": score} for score in scores]


class Reward(BulkInstanceMetric):
    reduction_map = {"mean": ["score"]}
    main_score = "score"
    batch_size: int = 32

    model_name: str

    prediction_type = "str"
    single_reference_per_prediction = True

    _requirements_list: List[str] = ["transformers", "torch"]

    def prepare(self):
        super().prepare()
        import torch
        from transformers import pipeline

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "text-classification", model=self.model_name, device=device
        )

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        # treat the references as the questions and the predictions as answers
        # assume a single reference
        questions = [refs[0] for refs in references]
        answers = predictions

        # prepare for computation
        inputs = [{"text": q, "text_pair": a} for q, a in zip(questions, answers)]

        # compute the metric
        # add function_to_apply="none" to disable sigmoid
        return self.pipe(inputs, batch_size=self.batch_size)


class Detector(BulkInstanceMetric):
    reduction_map = {"mean": ["score"]}
    main_score = "score"
    batch_size: int = 32

    prediction_type = "str"

    model_name: str

    _requirements_list: List[str] = ["transformers", "torch"]

    def prepare(self):
        super().prepare()
        import torch
        from transformers import pipeline

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "text-classification", model=self.model_name, device=device
        )

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        # compute the metric
        # add function_to_apply="none" to disable sigmoid
        return self.pipe(predictions, batch_size=self.batch_size)


class LlamaIndexLLMMetric(InstanceMetric):
    model_name: str = ""
    main_score: str = ""
    prediction_type: str = "str"
    reduction_map: Dict[str, List[str]] = None
    openai_models: List[str] = ["gpt-3.5-turbo"]
    anthropic_models: List[
        str
    ] = []  # this is here for the sake of documentation for future models
    mock_models: List[str] = ["mock"]
    external_api_models = openai_models + anthropic_models
    data_classification_policy = ["public"]

    _requirements_list: List[str] = ["llama_index"]

    def prepare(self):
        self.model_name_normalized = self.model_name.replace(".", "_").replace("-", "_")
        self.main_score: str = f"llama_index_by_{self.model_name_normalized}_judge"

        self.reduction_map: Dict[str, List[str]] = {"mean": [self.main_score]}

        if self.model_name in self.openai_models:
            from llama_index.llms.openai import OpenAI

            self.llm = OpenAI("gpt-3.5-turbo")
        elif self.model_name in self.mock_models:
            from llama_index.core.llms.mock import MockLLM

            self.llm = MockLLM(system_prompt="5")  # perfect score
        else:
            raise NotImplementedError(
                f"LlamaIndexLLM metric does not support {self.model_name}, currently only gpt-3.5-turbo is supported"
            )

    def _model_using_extrnal_api(self):
        return self.model_name in self.external_api_models


class LlamaIndexCorrectness(LlamaIndexLLMMetric):
    """LlamaIndex based metric class for evaluating correctness."""

    score_prefix = "correctness_"

    @staticmethod
    def _custom_parser(eval_response: str):
        """Default parser function for evaluation response.

        Args:
            eval_response (str): The response string from the evaluation.

        Returns:
            Tuple[float, str]: A tuple containing the score as a float and the reasoning as a string.
        """
        import re

        match = re.search(r"\b\d+\.\d+\b|\b\d+\b", eval_response)

        if match:
            score = float(match.group())
        else:
            raise Exception("could not parse judge response")

        reasoning_str = "\n".join(eval_response.split("\n")[1:])
        reasoning = reasoning_str.lstrip("\n")
        return score, reasoning

    def prepare(self):
        """Initialization method for the metric. Initializes the CorrectnessEvaluator with the OpenAI model."""
        super().prepare()

        from llama_index.core.evaluation import CorrectnessEvaluator

        self.evaluator = CorrectnessEvaluator(
            llm=self.llm, parser_function=self._custom_parser
        )

    def compute(
        self,
        references: List[str],
        prediction: str,
        task_data: Dict,
    ) -> Dict[str, Any]:
        """Method to compute the correctness metric.

        Args:
            references (List[str]): List of reference instances.
            prediction (str): List of predicted instances.
            task_data (Dict): List of additional input data.

        Returns:
            Dict[str, Any]: List of computed scores and feedback.

        Raises:
            AssertionError: If the input does not meet the expected format.
        """
        query = task_data["question"]

        contexts = None
        if "contexts" in task_data:
            contexts = task_data["contexts"]

        per_reference_results = []
        for reference_response in references:
            per_reference_results.append(
                self.evaluator.evaluate(
                    query=query,
                    response=prediction,
                    contexts=contexts,
                    reference=reference_response,
                )
            )
        result = max([results.score for results in per_reference_results])

        return {self.main_score: result / 5}


class LlamaIndexFaithfulness(LlamaIndexLLMMetric):
    """LlamaIndex based metric class for evaluating faithfulness."""

    score_prefix = "faithfulness_"

    def prepare(self):
        """Initialization method for the metric. Initializes the FaithfulnessEvaluator with the OpenAI model."""
        super().prepare()

        from llama_index.core.evaluation import FaithfulnessEvaluator

        self.evaluator = FaithfulnessEvaluator(llm=self.llm)

    def compute(
        self,
        references: List[str],
        prediction: str,
        task_data: Dict,
    ) -> Dict[str, Any]:
        result = self.evaluator.evaluate(
            query=task_data["question"],
            response=prediction,
            contexts=task_data["contexts"],
        )
        score = result.score

        return {self.main_score: score}


class Perplexity(BulkInstanceMetric):
    """Computes the likelihood of generating text Y after text X - P(Y|X)."""

    main_score = "perplexity"
    reduction_map = {"mean": ["perplexity"]}
    prediction_type = "str"

    source_template: str
    target_template: str
    batch_size: int = 32
    model_name: str
    single_token_mode: bool = False

    lm = None

    _requirements_list: List[str] = ["transformers", "torch"]

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Computes the likelihood of generating text Y after text X - P(Y|X).

        :param predictions: the list of Y texts = the targets of the generation
        :param references: the list of list of X texts = the sources of the generation

        :return: the likelihood of generating text Y_i after each text X_i_j = P(Y_i|X_i_1), ..., P(Y_i|X_i_n)  for every i.
        """
        if self.lm is None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self.lm = (
                self.EncoderDecoderLM(
                    model_name=self.model_name, single_token_mode=self.single_token_mode
                )
                if config.is_encoder_decoder is True
                else self.DecoderOnlyLM(
                    model_name=self.model_name, single_token_mode=self.single_token_mode
                )
            )

        sources = []
        targets = []
        for prediction, instance_references in zip(predictions, references):
            for instance_reference in instance_references:
                sources.append(
                    self.Template.apply(
                        self.source_template,
                        prediction=prediction,
                        reference=instance_reference,
                    )
                )
                targets.append(
                    self.Template.apply(
                        self.target_template,
                        prediction=prediction,
                        reference=instance_reference,
                    )
                )

        # compute P(Q|P) and store in queue
        scores = self.lm.compute_lm(
            source=sources, target=targets, batch_size=self.batch_size
        )

        index = 0
        all_instances_scores = []
        for instance_references in references:
            instance_scores = {}
            instance_scores_list = []
            for _ in range(len(instance_references)):
                instance_scores_list.append(scores[index])
                index += 1
            instance_scores["reference_scores"] = instance_scores_list

            # max seems more useful than mean for common use cases like
            # context relevance, where what we want to know is if there
            # is at least one good result in the context. Using mean will
            # bring the score down due to bad contexts at the tail.
            instance_scores[self.main_score] = max(instance_scores_list)
            all_instances_scores.append(instance_scores)

        return all_instances_scores

    class Template:
        regex = re.compile(r"\{(\w+)}")

        @classmethod
        def apply(cls, template, **kwargs):
            matches = Perplexity.Template.regex.finditer(template)
            output = []
            cursor = 0
            for match in matches:
                start = match.start()
                end = match.end()
                output.append(template[cursor:start])
                output.append(kwargs[match.group(1)])
                cursor = end
            output.append(template[cursor:])
            return "".join(output)

    class AbstractLM(ABC):
        def __init__(self, model_name, single_token_mode):
            import torch
            from transformers import AutoTokenizer

            self.model_name = model_name
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = (
                self.model_class().from_pretrained(self.model_name).to(self.device)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.single_token_mode = single_token_mode

        def compute_lm(
            self, source: List[str], target: List[str], batch_size: int
        ) -> List[float]:
            import torch

            scores = []

            with torch.no_grad():
                # break the documents to batches
                n_batches = int(len(source) / batch_size)
                batch_range = range(n_batches + 1)
                for batch in batch_range:
                    batch_source = source[batch * batch_size : (batch + 1) * batch_size]
                    batch_target = target[batch * batch_size : (batch + 1) * batch_size]
                    if len(batch_source) > 0:
                        # tokenize the source and target
                        tokens_source = self.tokenizer(
                            batch_source, padding=True, return_tensors="pt"
                        )
                        tokens_target = self.tokenizer(
                            batch_target,
                            padding=True,
                            return_tensors="pt",
                            add_special_tokens=not self.single_token_mode,
                        )

                        # compute the logits
                        logits, labels = self.compute_batch(
                            tokens_source, tokens_target
                        )

                        # logits is a tensor of size: batch_size * len(target) * vocab_size
                        # because for each example in the batch, the model predicted the
                        # logit at every position in the target, for every vocab item.

                        # the model returns mean over all batch. We run the CE again without reduction
                        # and extract the mean for each document
                        loss_fct = torch.nn.CrossEntropyLoss(
                            ignore_index=-100, reduction="none"
                        )

                        # logits.size(-1) = the dimension of the vocabulary
                        # labels.view(-1) = flattens the labels tensor to 1d
                        loss = loss_fct(
                            logits.view(-1, logits.size(-1)), labels.view(-1)
                        )
                        loss = loss.view(len(batch_source), -1)

                        # for each document, do mean only over the non zero values (sum(labels>0))
                        batch_loss = torch.sum(loss, dim=1) / torch.sum(
                            labels > 0, dim=1
                        )

                        # e^-average(cross-entropy-loss(logits) == geometric mean of the probabilities
                        # proof:
                        # * CE-loss of logits is computed by transforming the logits to
                        #   probabilities by softmax, and then -log(p) is returned, where
                        #   p is the probability of the gold label.
                        # * Averaging the CE loss is computed by summing over -log(p) and
                        #   then dividing by the length of the gold labels.
                        # * Thus, pr_score = (-log(p_1) +  ... + -log(p_n)) / n
                        #                  = -log(p_1 * ... * p_n) * 1/n
                        # * Therefore,
                        #   e^(-pr_score) = e^(log(p_1 * ... * p_n) * 1/n)
                        #                 = (e^(log(p_1 * ... * p_n))) ^ 1/n
                        #                 = p_1 * ... * p_n) ^ 1/n
                        #                 = geometric mean of [p_1, ..., p_n]
                        #
                        # in principle we could have computed the geometric mean directly over the
                        # probabilities instead of e^(average cross entropy loss of the logits),
                        # but the current approach is more stable numerically.  See for example:
                        # https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way
                        geometric_mean = (-batch_loss).exp()

                        # append the batch scores to the list of all scores
                        scores.append(geometric_mean)

            return torch.cat(scores, dim=0).tolist()

        @abstractmethod
        def model_class(self):
            pass

        @abstractmethod
        def compute_batch(self, tokens_source, tokens_target):
            pass

    class EncoderDecoderLM(AbstractLM):
        def model_class(self):
            from transformers import AutoModelForSeq2SeqLM

            return AutoModelForSeq2SeqLM

        def compute_batch(self, tokens_source, tokens_target):
            tokens_docs_ids = tokens_source["input_ids"].to(self.device)
            attention = tokens_source["attention_mask"].to(self.device)
            labels = tokens_target["input_ids"].to(self.device)

            logits = self.model(
                input_ids=tokens_docs_ids.long(),
                attention_mask=attention.long(),
                labels=labels.long(),
            ).logits

            # replace the padding token in the labels by -100
            labels[labels == self.tokenizer.pad_token_id] = -100

            return logits, labels

    class DecoderOnlyLM(AbstractLM):
        def model_class(self):
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

        def compute_batch(self, tokens_source, tokens_target):
            import torch

            tokens = torch.cat(
                [tokens_source["input_ids"], tokens_target["input_ids"]], dim=1
            )
            attention = torch.cat(
                [tokens_source["attention_mask"], tokens_target["attention_mask"]],
                dim=1,
            )
            labels = torch.cat(
                [
                    torch.zeros_like(tokens_source["input_ids"]).fill_(-100),
                    tokens_target["input_ids"],
                ],
                dim=1,
            )

            # replace the padding token in the labels by -100
            labels[labels == self.tokenizer.pad_token_id] = -100

            tokens = tokens.to(self.device)
            attention = attention.to(self.device)
            labels = labels.to(self.device)

            # no need to pass labels as we calculate the loss below per document
            model_output = self.model(
                input_ids=tokens.long(), attention_mask=attention.long()
            )
            logits = model_output.logits

            # in decoder only, the first token is not being generated, it is taken from the input,
            # so the model is generating from token 2 to n+1. therefore, we need to skip the last
            # logit and the first label.
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()

            return shifted_logits, shifted_labels


class Squad(HuggingfaceMetric):
    hf_metric_name = "squad"
    main_score = "f1"
    scale = 100.0
    scaled_fields = ["f1", "exact_match"]
    prediction_type = "Dict[str,Any]"

    # Squad references are not list, but a dict that contain a field called 'answers/text'
    # which is the list of references
    def _validate_reference(self, reference):
        if not isoftype(reference, self.get_prediction_type()):
            raise ValueError(
                f"Each reference is expected to be of type '{self.prediction_type}' in {self.get_metric_name()} metric. Received prediction of type {type(reference)}: {reference}"
            )


class NDCG(GlobalMetric):
    """Normalized Discounted Cumulative Gain: measures the quality of ranking with respect to ground truth ranking scores.

    As this measures ranking, it is a global metric that can only be calculated over groups of instances. In the
    common use case where the instances are grouped by different queries, i.e., where the task is to provide a
    relevance score for a search result w.r.t. a query, an nDCG score is calculated per each query (specified in the
    "query" input field of an instance) and the final score is the average across all queries.
    Note that the expected scores are relevance scores (i.e., higher is better) and not rank indices. The absolute
    value of the scores is only meaningful for the reference scores; for the predictions, only the ordering of the
    scores affects the outcome - for example, predicted scores of [80, 1, 2] and [0.8, 0.5, 0.6] will receive
    the same nDCG score w.r.t. a given set of reference scores.

    See also https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """

    main_score = "nDCG"

    _requirements_list: List[str] = ["sklearn"]
    single_reference_per_prediction = True
    prediction_type = "Optional[float]"

    def prepare(self):
        from sklearn.metrics import ndcg_score

        super().prepare()
        self.eval = ndcg_score

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Any],
    ) -> dict:
        from collections import defaultdict

        query_to_predictions_and_references = defaultdict(lambda: [[], []])
        references = [reference[0] for reference in references]
        for reference, pred, inputs_dict in zip(references, predictions, task_data):
            query = inputs_dict.get("query")
            query_to_predictions_and_references[query][0].append(pred)
            query_to_predictions_and_references[query][1].append(reference)

        scores = []
        for q_predictions, q_references in query_to_predictions_and_references.values():
            if len(q_references) == 1:
                continue

            if (
                None in q_predictions
            ):  # model failed to predict numeric scores for some instances
                numeric_predictions = [
                    pred for pred in q_predictions if pred is not None
                ]
                if len(numeric_predictions) <= 1:  # no meaningful ranking
                    scores.append(0)
                    continue
                # consider non-numeric model predictions as ranked last
                min_value = min(numeric_predictions)
                q_predictions = [
                    1 + (pred - min_value) if pred is not None else 0
                    for pred in q_predictions
                ]
            scores.append(self.eval([q_references], [q_predictions]))
        return {self.main_score: mean(scores) if len(scores) > 0 else np.nan}


class RetrievalMetric(InstanceMetric):
    prediction_type = "List[str]"
    single_reference_per_prediction = True

    def compute(self, references: List[Any], prediction: Any, task_data: Dict) -> dict:
        # digest input
        pred_ids: List[Any] = prediction
        ref_ids: List[Any] = list(dict.fromkeys(references[0]))

        # relevance_at_k: 1-based dictionary of indicators (0/1), telling whether
        # the doc id retrieved at position k (assuming it is 1-based, so k starts
        # from 1) is in the gold doc ids or not.
        # For example, assuming that in the retrieved docs we have correct predictions
        # at positions 2, 4 and 5 (1-based), the dict will look like:
        # {1: 0, 2: 1, 3: 0, 4: 1, 5: 1, ...}
        relevance_at_k = {
            k + 1: 1 if doc_id in ref_ids else 0 for k, doc_id in enumerate(pred_ids)
        }

        # relevance_sum_at_k: 1-based dictionary of counts, where the value at k determines
        # how many gold doc ids have been observed up to index k.
        relevance_sum_at_k = {}
        for k, value in relevance_at_k.items():
            relevance_sum_at_k[k] = relevance_sum_at_k.get(k - 1, 0) + value

        # precision_at_k: the precision of the top k retrieved documents. For example,
        # assuming that only 1 out of the first 4 retrieved documents is correct, the
        # value at 4 will be 1/4.
        precision_at_k = {k: value / k for k, value in relevance_sum_at_k.items()}

        # recall_at_k: the recall of the top k retrieved documents. For example,
        # assuming that only 2 out of the 3 gold documents are in the top 5 results,
        # the value at 5 will be 2/3.
        n_refs = len(ref_ids)
        recall_at_k = {
            k: value / n_refs if n_refs > 0 else 0
            for k, value in relevance_sum_at_k.items()
        }

        # rank - the 1-based index of the first hit of a gold doc id. So 1
        # means first position.
        rank = 0
        for k, relevance in relevance_at_k.items():
            if relevance == 1:
                rank = k
                break

        # match_at_k: whether we have a match at the top k retrieved documents
        match_at_k = {
            k: 1.0 if value > 0 else 0.0 for k, value in relevance_sum_at_k.items()
        }

        return self._compute(
            relevance_at_k,
            relevance_sum_at_k,
            precision_at_k,
            recall_at_k,
            match_at_k,
            rank,
        )

    @abstractmethod
    def _compute(
        self,
        relevance_at_k,
        relevance_sum_at_k,
        precision_at_k,
        recall_at_k,
        match_at_k,
        rank,
    ) -> dict:
        pass


class MRR(RetrievalMetric):
    reduction_map = {"mean": ["mrr"]}
    main_score = "mrr"
    ci_scores = ["mrr"]

    def _compute(
        self,
        relevance_at_k,
        relevance_sum_at_k,
        precision_at_k,
        recall_at_k,
        match_at_k,
        rank,
    ) -> dict:
        return {self.main_score: 1 / rank if rank > 0 else 0}


class MAP(RetrievalMetric):
    reduction_map = {"mean": ["map"]}
    main_score = "map"
    ci_scores = ["map"]

    def _compute(
        self,
        relevance_at_k,
        relevance_sum_at_k,
        precision_at_k,
        recall_at_k,
        match_at_k,
        rank,
    ) -> dict:
        result = 0
        if len(relevance_at_k) > 0:
            total = sum(relevance_at_k.values())
            if total > 0:
                dot = sum(relevance_at_k[k] * precision_at_k[k] for k in relevance_at_k)
                result = dot / total
        return {self.main_score: result}


class RetrievalAtK(RetrievalMetric):
    k_list: List[int]
    main_score: str = None
    reduction_map: Dict[str, List[str]] = None

    def prepare(self):
        super().prepare()
        self.main_score = self.score_name("match", self.k_list[0])
        self.ci_scores = [
            self.score_name(measure, k)
            for measure in ["precision", "recall", "match"]
            for k in self.k_list
        ]
        self.reduction_map = {"mean": self.ci_scores}

    @staticmethod
    def score_name(measure: str, k: int):
        return f"{measure}_at_{k}"

    def _compute(
        self,
        relevance_at_k,
        relevance_sum_at_k,
        precision_at_k,
        recall_at_k,
        match_at_k,
        rank,
    ) -> dict:
        result = {}
        for measure_array, measure_name in [
            (precision_at_k, "precision"),
            (recall_at_k, "recall"),
            (match_at_k, "match"),
        ]:
            max_k = max(measure_array.keys())
            for k in self.k_list:
                result[self.score_name(measure_name, k)] = measure_array[min(k, max_k)]
        return result


class KPA(CustomF1):
    prediction_type = "str"
    single_reference_per_prediction = True

    def get_element_group(self, element, additional_input):
        return additional_input["keypoint"]

    def get_element_representation(self, element, additional_input):
        return additional_input["keypoint"]

    def should_ignore_element(self, element, additional_input):
        return element == "none"


class RemoteMetric(StreamOperator, Metric):
    """A metric that runs another metric remotely.

    main_score: the score updated by this metric.
    endpoint: the remote host that supports the remote metric execution.
    metric_name: the name of the metric that is executed remotely.
    api_key: optional, passed to the remote metric with the input, allows secure authentication.
    """

    main_score: str = None
    endpoint: str
    metric_name: str
    api_key: str = None
    data_classification_policy = ["public", "proprietary"]

    @staticmethod
    def wrap_inner_metric_pipeline_metric(
        metric_pipeline: MetricPipeline,
        remote_metrics_endpoint: str,
    ) -> MetricPipeline:
        """Wrap the inner metric in a MetricPipeline with a RemoteMetric.

        When executing the returned MetricPipeline, the inner metric will be computed
        remotely (pre and post processing steps in the MetricPipeline will be computed locally).
        """
        local_inner_metric = metric_pipeline.metric
        metric_pipeline = deepcopy(
            metric_pipeline
        )  # To avoid unintentional changes to the catalog contents
        metric_pipeline.metric = RemoteMetric(
            main_score=local_inner_metric.main_score,
            metric_name=local_inner_metric.__id__,
            endpoint=remote_metrics_endpoint,
        )
        return metric_pipeline

    def get_metric_url(self) -> str:
        return f"{self.endpoint}/{self.metric_name}"

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        predictions, references, additional_inputs, instances = self.consume_stream(
            stream
        )
        metric_request = self.create_metric_request(
            predictions, references, additional_inputs
        )
        metric_response = self.get_metric_response(metric_request)
        self.update_instance_scores(instances, metric_response.instances_scores)
        self.set_global_score(instances, metric_response.global_score)
        yield from instances

    @staticmethod
    def create_metric_request(predictions, references, additional_inputs):
        instance_inputs = [
            InstanceInput(
                prediction=prediction,
                references=reference,
                additional_inputs=additional_input,
            )
            for prediction, reference, additional_input in zip(
                predictions, references, additional_inputs
            )
        ]
        return MetricRequest(instance_inputs=instance_inputs)

    def get_metric_response(self, metric_request: MetricRequest) -> MetricResponse:
        import requests

        response = requests.post(
            url=self.get_metric_url(),
            json=metric_request.to_dict(),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        response_json = response.json()
        return MetricResponse(**response_json)

    def disable_confidence_interval_calculation(self):
        """Confidence intervals are always disabled for RemoteMetric.

        No need to do anything.
        """
        pass

    def set_n_resamples(self, n_resample):
        """Since confidence intervals are always disabled for remote metrics, this is a no-op."""
        pass


def validate_subgroup_types(
    subgroup_scores_dict: Dict[str, List],
    control_subgroup_types: List[str],
    comparison_subgroup_types: List[str],
):
    """Validate a dict of subgroup type instance score lists, and subgroup type lists.

    Args:
        subgroup_scores_dict: dict where keys are subgroup types and values are lists of instance scores.
        control_subgroup_types: list of subgroup types (potential keys of subgroup_scores_dict) that are the control (baseline) group
        comparison_subgroup_types: list of subgroup types (potential keys of subgroup_scores_dict) that are the group
            to be compared to the control group.

    Returns:
        dict with all NaN scores removed; control_subgroup_types and comparison_subgroup_types will have non-unique elements removed
    """
    # note: subgroup_scores_dict is already a defaultdict of lists, so don't need to check that keys in control_ and comparison_subgroup_types exist in it
    # remove any NaNs
    subgroup_scores_dict.update(
        {
            subgroup_name: [score for score in score_list if not np.isnan(score)]
            for subgroup_name, score_list in subgroup_scores_dict.items()
        }
    )
    assert isinstance(
        control_subgroup_types, list
    ), "control_subgroup_types must be a list"
    assert isinstance(
        comparison_subgroup_types, list
    ), "comparison_subgroup_types must be a list"
    # make sure each list is unique, so that labels aren't double-counted
    control_subgroup_types = list(set(control_subgroup_types))
    comparison_subgroup_types = list(set(comparison_subgroup_types))

    return subgroup_scores_dict, control_subgroup_types, comparison_subgroup_types


def performance_drop_rate(
    subgroup_scores_dict: Dict[str, List],
    control_subgroup_types: List[str],
    comparison_subgroup_types: List[str],
):
    """Percentage decrease of mean performance on test elements relative to that on a baseline (control).

    from https://arxiv.org/pdf/2306.04528.pdf.

    Args:
        subgroup_scores_dict: dict where keys are subgroup types and values are lists of instance scores.
        control_subgroup_types: list of subgroup types (potential keys of subgroup_scores_dict) that are the control (baseline) group
        comparison_subgroup_types: list of subgroup types (potential keys of subgroup_scores_dict) that are the group
            to be compared to the control group.

    Returns:
        numeric PDR metric.
        If only one element (no test set) or the first is 0 (percentage change is undefined) return NaN
        otherwise, calculate PDR
    """
    (
        subgroup_scores_dict,
        control_subgroup_types,
        comparison_subgroup_types,
    ) = validate_subgroup_types(
        subgroup_scores_dict, control_subgroup_types, comparison_subgroup_types
    )

    # combine all scores from each label (if there are more than 1 in each group) into a list
    group_scores_list = [
        np.concatenate(
            [subgroup_scores_dict[subgroup_name] for subgroup_name in name_list]
        )
        for name_list in [control_subgroup_types, comparison_subgroup_types]
    ]
    if any(len(scores) == 0 for scores in group_scores_list):
        # no comparison can be made since there is not at least one score per type
        return np.nan
    control_mean = mean(group_scores_list[0])
    comparison_mean = mean(group_scores_list[1])
    if control_mean == 0:
        # return 0 if comparison is also 0
        if comparison_mean == 0:
            return 0
        return np.nan
    # otherwise, take the percentage change (which may also be 0)
    return 1 - comparison_mean / control_mean


def interpret_effect_size(x: float):
    """Return a string rule-of-thumb interpretation of an effect size value, as defined by Cohen/Sawilowsky.

    See https://en.wikipedia.org/wiki/Effect_size;
    Cohen, Jacob (1988). Statistical Power Analysis for the Behavioral Sciences; and
    Sawilowsky, S (2009). "New effect size rules of thumb". Journal of Modern Applied Statistical Methods. 8 (2): 467-474.

    Value has interpretation of
    - essentially 0 if |x| < 0.01
    - very small if 0.01 <= |x| < 0.2
    - small difference if 0.2 <= |x| < 0.5
    - a medium difference if 0.5 <= |x| < 0.8
    - a large difference if 0.8 <= |x| < 1.2
    - a very large difference if 1.2 <= |x| < 2.0
    - a huge difference if 2.0 <= |x|

    Args:
        x: float effect size value

    Returns:
        string interpretation
    """
    import pandas as pd

    # assign a label according to threshold of the absolute value
    return pd.cut(
        x=[np.abs(x)],
        right=False,
        bins=[-1, 0.01, 0.2, 0.5, 0.8, 1.2, 2.0, np.Inf],
        labels=[
            "essentially zero",
            "very small",
            "small",
            "medium",
            "large",
            "very large",
            "huge",
        ],
    )[0]


def normalized_cohens_h(
    subgroup_scores_dict: Dict[str, List],
    control_subgroup_types: List[str],
    comparison_subgroup_types: List[str],
    interpret=False,
):
    """Cohen's h effect size between two proportions, normalized to interval [-1,1].

    Allows for change-type metric when the baseline is 0 (percentage change, and thus PDR, is undefined)
    https://en.wikipedia.org/wiki/Cohen%27s_h

    Cohen's h effect size metric between two proportions p2 and p1 is 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1))).
    h in -pi, pi, with +/-pi representing the largest increase/decrease (p1=0, p2=1), or (p1=1, p2=0).
    h=0 is no change. Unlike percentage change, h is defined even if the baseline (p1) is 0.
    Assumes the scores are in [0,1], either continuous or binary; hence taking the average of a group of scores yields a proportion..
    Calculates the change in the average of the other_scores relative to the average of the baseline_scores.    We rescale this to [-1,1] from [-pi,pi] for clarity, where +- 1 are the most extreme changes, and 0 is no change

    Interpretation: the original unscaled Cohen's h can be interpreted according to function interpret_effect_size

    Thus, the rule of interpreting the effect of the normalized value is to use the same thresholds divided by pi
        - essentially 0 if |norm h| < 0.0031831
        - very small if 0.0031831 <= |norm h| < 0.06366198
        - small difference if 0.06366198 <= |norm h| < 0.15915494
        - a medium difference if 0.15915494 <= |norm h| < 0.25464791
        - a large difference if 0.25464791 <= |norm h| < 0.38197186
        - a very large difference if 0.38197186 <= |norm h| < 0.63661977
        - a huge difference if 0.63661977 <= |norm h|
    Args:
        subgroup_scores_dict: dict where keys are subgroup types and values are lists of instance scores.
        control_subgroup_types: list of subgroup types (potential keys of subgroup_scores_dict) that are the control (baseline) group
        comparison_subgroup_types: list of subgroup types (potential keys of subgroup_scores_dict) that are the group
            to be compared to the control group.
        interpret: boolean, whether to interpret the significance of the score or not
    Returns:
        float score between -1 and 1, and a string interpretation if interpret=True
    """
    (
        subgroup_scores_dict,
        control_subgroup_types,
        comparison_subgroup_types,
    ) = validate_subgroup_types(
        subgroup_scores_dict, control_subgroup_types, comparison_subgroup_types
    )

    # requires scores to be in [0,1]
    for subgroup_name, score_list in subgroup_scores_dict.items():
        assert all(
            0 <= score <= 1 for score in score_list
        ), f"all {subgroup_name} scores must be in [0,1]"

    # combine all scores from each label (if there are more than 1 in each group) into a list
    group_scores_list = [
        np.concatenate(
            [subgroup_scores_dict[subgroup_name] for subgroup_name in name_list]
        )
        for name_list in [control_subgroup_types, comparison_subgroup_types]
    ]

    if any(len(scores) == 0 for scores in group_scores_list):
        # no comparison can be made since there is not at least one score per type
        h, norm_h = np.nan, np.nan
    else:
        control_mean = mean(group_scores_list[0])
        comparison_mean = mean(group_scores_list[1])
        h = 2 * (np.arcsin(np.sqrt(comparison_mean)) - np.arcsin(np.sqrt(control_mean)))
        norm_h = np.clip(a=h / np.pi, a_min=-1, a_max=1)

    if not interpret:
        return norm_h

    return norm_h, interpret_effect_size(h)


def normalized_hedges_g(
    subgroup_scores_dict: Dict[str, List[float]],
    control_subgroup_types: List[str],
    comparison_subgroup_types: List[str],
    interpret=False,
):
    """Hedge's g effect size between mean of two samples, normalized to interval [-1,1].  Better than Cohen's d for small sample sizes.

    Takes into account the variances within the samples, not just the means.

    Args:
        subgroup_scores_dict: dict where keys are subgroup types and values are lists of instance scores.
        control_subgroup_types: list of subgroup types (potential keys of subgroup_scores_dict) that are the control (baseline) group
        comparison_subgroup_types: list of subgroup types (potential keys of subgroup_scores_dict) that are the group
            to be compared to the control group.
        interpret: boolean, whether to interpret the significance of the score or not
    Returns:
        float score between -1 and 1, and a string interpretation if interpret=True
    """
    (
        subgroup_scores_dict,
        control_subgroup_types,
        comparison_subgroup_types,
    ) = validate_subgroup_types(
        subgroup_scores_dict, control_subgroup_types, comparison_subgroup_types
    )

    # combine all scores from each label (if there are more than 1 in each group) into a list
    group_scores_list = [
        np.concatenate(
            [subgroup_scores_dict[subgroup_name] for subgroup_name in name_list]
        )
        for name_list in [control_subgroup_types, comparison_subgroup_types]
    ]

    group_n = [len(scores) for scores in group_scores_list]
    if any(nn == 0 for nn in group_n) or all(nn <= 1 for nn in group_n):
        # if at least one sample size is 0 for one type, no comparison can be made at all
        # if both sample sizes are 1, then the denominator is undefined since divide by n1 + n2 - 2
        # so require at least one sample to have > 1 observation, and both to have >= 1.
        g, norm_g = np.nan, np.nan
    else:
        # otherwise, calculate the variances
        group_mean = [mean(scores) for scores in group_scores_list]
        # sample variance with 1 degree of freedom (denominator n-1); if n=1, return 0 since otherwise throws an error
        group_var = [
            0.0 if nn == 1 else np.var(scores, ddof=1)
            for scores, nn in zip(group_scores_list, group_n)
        ]
        var_total = sum([(nn - 1) * vv for vv, nn in zip(group_var, group_n)])
        pooled_sd = np.sqrt(var_total / (sum(group_n) - 2))

        max_absolute_value = 5
        gmd = float(group_mean[1] - group_mean[0])

        if gmd == 0:
            # if exactly the same, return 0
            g = 0.0
        else:
            try:
                g = gmd / pooled_sd
            except ZeroDivisionError:
                # return a large effect size to avoid explosion if there is zero variance
                g = np.sign(gmd) * max_absolute_value

        n = sum(group_n)
        if 3 < n < 50:
            # small sample adjustment see https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
            # the multiplier is 0 if n <= 3
            g *= ((n - 3) / (n - 2.25)) * np.sqrt((n - 2) / n)
        # clip it at a very large value so it doesn't become infinite if the variance (denominator) is very small or 0
        g = float(np.clip(a=g, a_min=-1 * max_absolute_value, a_max=max_absolute_value))
        norm_g = g / max_absolute_value

    if not interpret:
        return norm_g
    return norm_g, interpret_effect_size(g)


def mean_subgroup_score(
    subgroup_scores_dict: Dict[str, List], subgroup_types: List[str]
):
    """Return the mean instance score for a subset (possibly a single type) of variants (not a comparison).

    Args:
        subgroup_scores_dict: dict where keys are subgroup types and values are lists of instance scores.
        subgroup_types: the keys (subgroup types) for which the average will be computed.

    Returns:
        float score
    """
    subgroup_scores_dict, subgroup_types, _ = validate_subgroup_types(
        subgroup_scores_dict, subgroup_types, []
    )

    # combine all desired subgroup scores
    score_list = np.concatenate(
        [subgroup_scores_dict[subgroup_name] for subgroup_name in subgroup_types]
    )
    if len(score_list) == 0:
        # no scores to use
        return np.nan
    return mean(score_list)


# metrics using mean reduction
class GroupMeanAccuracy(Accuracy):
    reduction_map = {"group_mean": {"agg_func": ["mean", nan_mean, False]}}


class FixedGroupMeanAccuracy(Accuracy):
    # the same as GroupMeanAccuracy, except the groups are fixed and are resampled together
    reduction_map = {"group_mean": {"agg_func": ["mean", nan_mean, True]}}


# same as above, now using StringContainment
class GroupMeanStringContainment(StringContainment):
    reduction_map = {"group_mean": {"agg_func": ["mean", nan_mean, False]}}


class FixedGroupMeanStringContainment(StringContainment):
    # the same as GroupMeanStringContainment, except the groups are fixed and are resampled together
    reduction_map = {"group_mean": {"agg_func": ["mean", nan_mean, True]}}


# take only the (fixed) group mean of baseline or other (paraphrases) scores
class FixedGroupMeanBaselineAccuracy(Accuracy):
    subgroup_column = "variant_type"
    # take mean of "original" variants only
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "mean_baseline",
                lambda scd: mean_subgroup_score(
                    subgroup_scores_dict=scd, subgroup_types=["original"]
                ),
                True,
            ],
        }
    }


class FixedGroupMeanParaphraseAccuracy(Accuracy):
    subgroup_column = "variant_type"
    # take mean of "paraphrase" variants only
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "mean_paraphrase",
                lambda scd: mean_subgroup_score(
                    subgroup_scores_dict=scd, subgroup_types=["paraphrase"]
                ),
                True,
            ],
        }
    }


# same as above but using StringContainment
class FixedGroupMeanBaselineStringContainment(StringContainment):
    subgroup_column = "variant_type"
    # take mean of "original" variants only
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "mean_baseline",
                lambda scd: mean_subgroup_score(
                    subgroup_scores_dict=scd, subgroup_types=["original"]
                ),
                True,
            ],
        }
    }


class FixedGroupMeanParaphraseStringContainment(StringContainment):
    subgroup_column = "variant_type"
    # take mean of "paraphrase" variants only
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "mean_paraphrase",
                lambda scd: mean_subgroup_score(
                    subgroup_scores_dict=scd, subgroup_types=["paraphrase"]
                ),
                True,
            ],
        }
    }


# using PDR
class FixedGroupPDRParaphraseAccuracy(Accuracy):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "pdr_paraphrase",
                lambda scd: performance_drop_rate(
                    subgroup_scores_dict=scd,
                    control_subgroup_types=["original"],
                    comparison_subgroup_types=["paraphrase"],
                ),
                True,
            ],
        }
    }


class FixedGroupPDRParaphraseStringContainment(StringContainment):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "pdr_paraphrase",
                lambda scd: performance_drop_rate(
                    subgroup_scores_dict=scd,
                    control_subgroup_types=["original"],
                    comparison_subgroup_types=["paraphrase"],
                ),
                True,
            ],
        }
    }


class GroupMeanTokenOverlap(TokenOverlap):
    reduction_map = {
        "group_mean": {
            "agg_func": ["mean", nan_mean, False],
            "score_fields": ["f1", "precision", "recall"],
        }
    }


# using Cohens's h for proportions
class FixedGroupNormCohensHParaphraseAccuracy(Accuracy):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "norm_cohens_h_paraphrase",
                lambda scd: normalized_cohens_h(
                    subgroup_scores_dict=scd,
                    control_subgroup_types=["original"],
                    comparison_subgroup_types=["paraphrase"],
                ),
                True,
            ],
        }
    }


class FixedGroupNormCohensHParaphraseStringContainment(StringContainment):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "norm_cohens_h_paraphrase",
                lambda scd: normalized_cohens_h(
                    subgroup_scores_dict=scd,
                    control_subgroup_types=["original"],
                    comparison_subgroup_types=["paraphrase"],
                ),
                True,
            ],
        }
    }


# using Hedges' g (takes into account internal variation in group scores)
class FixedGroupNormHedgesGParaphraseAccuracy(Accuracy):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "norm_hedges_g_paraphrase",
                lambda scd: normalized_hedges_g(
                    subgroup_scores_dict=scd,
                    control_subgroup_types=["original"],
                    comparison_subgroup_types=["paraphrase"],
                ),
                True,
            ],
        }
    }


class FixedGroupNormHedgesGParaphraseStringContainment(StringContainment):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "norm_hedges_g_paraphrase",
                lambda scd: normalized_hedges_g(
                    subgroup_scores_dict=scd,
                    control_subgroup_types=["original"],
                    comparison_subgroup_types=["paraphrase"],
                ),
                True,
            ],
        }
    }


# for above metrics, take absolute value of group score first; this measures variation in either direction
class FixedGroupAbsvalNormCohensHParaphraseAccuracy(Accuracy):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "absval_norm_cohens_h_paraphrase",
                lambda scd: np.abs(
                    normalized_cohens_h(
                        subgroup_scores_dict=scd,
                        control_subgroup_types=["original"],
                        comparison_subgroup_types=["paraphrase"],
                    )
                ),
                True,
            ],
        }
    }


class FixedGroupAbsvalNormCohensHParaphraseStringContainment(StringContainment):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "absval_norm_cohens_h_paraphrase",
                lambda scd: np.abs(
                    normalized_cohens_h(
                        subgroup_scores_dict=scd,
                        control_subgroup_types=["original"],
                        comparison_subgroup_types=["paraphrase"],
                    )
                ),
                True,
            ],
        }
    }


class FixedGroupAbsvalNormHedgesGParaphraseAccuracy(Accuracy):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "absval_norm_hedges_g_paraphrase",
                lambda scd: np.abs(
                    normalized_hedges_g(
                        subgroup_scores_dict=scd,
                        control_subgroup_types=["original"],
                        comparison_subgroup_types=["paraphrase"],
                    )
                ),
                True,
            ],
        }
    }


class FixedGroupAbsvalNormHedgesGParaphraseStringContainment(StringContainment):
    subgroup_column = "variant_type"
    reduction_map = {
        "group_mean": {
            "agg_func": [
                "absval_norm_hedges_g_paraphrase",
                lambda scd: np.abs(
                    normalized_hedges_g(
                        subgroup_scores_dict=scd,
                        control_subgroup_types=["original"],
                        comparison_subgroup_types=["paraphrase"],
                    )
                ),
                True,
            ],
        }
    }


class BinaryMaxF1(F1Binary):
    """Calculate the maximal F1 and the decision threshold that achieves it for a binary task with float predictions."""

    main_score = "max_f1_binary"
    single_reference_per_prediction = True
    average = None

    def compute(
        self,
        references: List[List[float]],
        predictions: List[List[float]],
        task_data: List[Dict],
    ) -> dict:
        best_thr = -1
        best_f1 = defaultdict(lambda: -1)
        best_thr_neg = -1
        best_f1_neg = defaultdict(lambda: -1)
        thrs = {round(fp, 3) for fp in predictions}
        for thr in thrs:
            new_predictions = [
                1.0 if float_prediction >= thr else 0.0
                for float_prediction in predictions
            ]
            f1_results = super().compute(references, new_predictions, task_data)

            f1 = f1_results["f1_binary"]
            if f1 > best_f1["f1_binary"]:
                best_f1 = f1_results.copy()
                best_thr = thr

            f1_neg = f1_results["f1_binary_neg"]
            if f1_neg > best_f1_neg["f1_binary_neg"]:
                best_f1_neg = f1_results.copy()
                best_thr_neg = thr

        return {
            self.main_score: best_f1["f1_binary"],
            "best_thr_maxf1": best_thr,
            f"{self.main_score}_neg": best_f1_neg["f1_binary_neg"],
            "best_thr_maxf1_neg": best_thr_neg,
            "recall_at_max_f1": best_f1["recall_binary"],
            "recall_at_max_f1_neg": best_f1_neg["recall_binary_neg"],
            "precision_at_max_f1": best_f1["precision_binary"],
            "precision_at_max_f1_neg": best_f1_neg["precision_binary_neg"],
        }


class BinaryAccuracy(InstanceMetric):
    """Calculate accuracy for a binary task, using 0.5 as the threshold in the case of float predictions."""

    reduction_map = {"mean": ["accuracy_binary"]}
    main_score = "accuracy_binary"
    ci_scores = ["accuracy_binary"]
    threshold = 0.5

    prediction_type = "Union[float,int]"
    single_reference_per_prediction = True

    def _validate_reference(self, reference):
        super()._validate_reference(reference)
        assert reference[0] in [
            0,
            1,
        ], f"all references of {self.main_score} must by 0 or 1"

    def compute(
        self, references: List[float], prediction: float, task_data: List[Dict]
    ) -> dict:
        prediction = int(prediction > self.threshold)
        reference = int(references[0])

        result = {self.main_score: float(prediction == reference)}
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result


class BinaryMaxAccuracy(GlobalMetric):
    """Calculate the maximal accuracy and the decision threshold that achieves it for a binary task with float predictions."""

    process_single_instances = False
    main_score = "max_accuracy_binary"
    prediction_type = "Union[float,int]"
    single_reference_per_prediction = True

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict],
    ) -> dict:
        references = [[int(r[0])] for r in references]

        # Sticking to the test >= thr, accuracy induced by threshold thr is the number of float predictions
        # that pass the test (are >= thr) and are paired with reference "1" plus the number of float predictions that
        # fail the test (are < thr) and are paired with reference "0".
        # A given threshold thr induces the same partition over the float predictions into passing and failing
        # as threshold thr' induces, with thr' being the smallest among the ones passing the test of thr.
        # Hence, we only need to review thresholds being float predictions, plus a threshold being larger than
        # the largest float predictions, to induce the partition into all-failing , none-passing.

        fp = [
            (predictions[i], i, -1 if references[i][0] == 1 else +1)
            for i in range(len(predictions))
        ]
        fp.sort()
        # each triplet above: float-prediction f; f's ordinal position in float_predictions, which is also
        # a means to obtain distinct triplets; and: the change in number of predictions that the test sends
        # to the reference they are paired with, a change implied by a move of thr that transfers f
        # from the set of passing the test to the set of failing it.

        rightmost_thr = 1.0 if fp[-1][0] < 1 else fp[-1][0] + 0.01
        # trying to be esthetic, have the threshold within [0,1], although this is not a requirement,
        # and even the float predictions are not guaranteed to be within the range [0,1]

        current_thr = fp[0][0]
        # partition float_predictions into all-passing, none-failing
        current_acc = sum(r[0] == 1 for r in references)
        # number of predictions that thr sends to the reference they are paired with

        best_acc = current_acc
        best_thr = current_thr

        i = 0
        while (i < len(predictions)) and (best_acc < len(predictions)):
            # best_acc can not exceed len(predictions)
            delta = fp[i][2]
            i += 1
            while i < len(predictions) and fp[i][0] <= fp[i - 1][0]:
                delta += fp[i][2]
                i += 1
            current_acc += delta
            if current_acc > best_acc:
                best_acc = current_acc
                best_thr = fp[i][0] if i < len(predictions) else rightmost_thr

        return {
            self.main_score: float(best_acc) / len(predictions),
            "best_thr_max_acc": best_thr,
        }


######################
# RerankRecallMetric #


def pytrec_eval_at_k(results, qrels, at_k, metric_name):
    import pandas as pd
    import pytrec_eval

    metric = {}

    for k in at_k:
        metric[f"{metric_name}@{k}"] = 0.0

    metric_string = f"{metric_name}." + ",".join([str(k) for k in at_k])
    # print('metric_string = ', metric_string)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg", metric_string}
    )  # {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)
    scores = pd.DataFrame(scores).transpose()

    keys = []
    column_map = {}
    for k in at_k:
        keys.append(f"{metric_name}_{k}")
        column_map[f"{metric_name}_{k}"] = k
    scores[keys].rename(columns=column_map)

    return scores


class RerankRecall(GlobalMetric):
    """RerankRecall: measures the quality of reranking with respect to ground truth ranking scores.

    This metric measures ranking performance across a dataset.  The
    references for a query will have a score of 1 for the gold passage
    and 0 for all other passages.  The model returns scores in [0,1]
    for each passage,query pair.  This metric measures recall at k by
    testing that the predicted score for the gold passage,query pair
    is at least the k'th highest for all passages for that query.  A
    query receives 1 if so, and 0 if not.  The 1's and 0's are
    averaged across the dataset.

    query_id_field selects the field containing the query id for an instance.
    passage_id_field selects the field containing the passage id for an instance.
    at_k selects the value of k used to compute recall.

    """

    main_score = "recall_at_5"
    query_id_field: str = "query_id"
    passage_id_field: str = "passage_id"
    at_k: List[int] = [1, 2, 5]

    # This doesn't seem to make sense
    n_resamples = None

    _requirements_list: List[str] = ["pandas", "pytrec_eval"]

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict],
    ):
        # Collect relevance score and ref per query/passage pair
        results = {}
        qrels = {}
        for ref, pred, data in zip(references, predictions, task_data):
            qid = data[self.query_id_field]
            pid = data[self.passage_id_field]
            if qid not in results:
                results[qid] = {}
                qrels[qid] = {}
            # Convert string-wrapped float to regular float
            try:
                results[qid][pid] = float(pred)
            except ValueError:
                # Card testing feeds nonnumeric values in, so catch that.
                results[qid][pid] = np.nan

            # There's always a single reference per pid/qid pair
            qrels[qid][pid] = int(ref[0])

        # Compute recall @ 5
        scores = pytrec_eval_at_k(results, qrels, self.at_k, "recall")
        # print(scores.describe())
        # pytrec returns numpy float32
        return {
            f"recall_at_{i}": float(scores[f"recall_{i}"].mean()) for i in self.at_k
        }


KO_ERROR_MESSAGE = """

Additional dependencies required. To install them, run:
`pip install "sacrebleu[ko]"`.

For MacOS: If error on 'mecab-config' show up during installation ], one should run:

`brew install mecab`
`pip install "sacrebleu[ko]"`

"""


class NormalizedSacrebleu(HuggingfaceMetric):
    hf_metric_name = "sacrebleu"
    hf_main_score = "score"
    prediction_type = "str"
    main_score = "sacrebleu"
    scale = 100.0
    scaled_fields = ["sacrebleu", "precisions"]
    hf_additional_input_fields_pass_one_value = ["tokenize"]
    _requirements_list = {
        "mecab_ko": KO_ERROR_MESSAGE,
        "mecab_ko_dic": KO_ERROR_MESSAGE,
    }


class CustomF1Fuzzy(CustomF1):
    def calculate_groups_ratio(self, actual_group, total_group):
        from fuzzywuzzy import fuzz

        tmp = []
        for actual_key in actual_group.keys():
            max_score = self.fuzz_ratio
            best_total_key = None

            for total_key in total_group.keys():
                tup_ac = ast.literal_eval(actual_key)
                tup_to = ast.literal_eval(total_key)

                if tup_ac[1] == tup_to[1]:
                    score = fuzz.ratio(tup_ac[0], tup_to[0])
                    if score > max_score:
                        max_score = score
                        best_total_key = total_key

            if best_total_key is not None:
                tmp.append(min(actual_group[actual_key], total_group[best_total_key]))
            else:
                tmp.append(min(actual_group[actual_key], 0))
        return sum(tmp), sum(actual_group.values())


class FuzzyNer(CustomF1Fuzzy):
    prediction_type = "List[Tuple[str,str]]"
    fuzz_ratio = 75

    def get_element_group(self, element, additional_input):
        return element[1]

    def get_element_representation(self, element, additional_input):
        return str(element)


class IsCodeMixed(BulkInstanceMetric):
    """Uses a generative model to assess whether a given text is code-mixed.

    Our goal is to identify whether a text is code-mixed, i.e., contains a mixture of different
    languages.
    The model is asked to identify the language of the text; if the model response begins with
    a number we take this as an indication that the text is code-mixed, for example:
    - Model response: "The text is written in 2 different languages"
    vs.
    - Model response: "The text is written in German"

    Note that this metric is quite tailored to specific model-template combinations, as it relies on the assumption
    that the model will complete the answer prefix "The text is written in ___" in a particular way.

    """

    main_score = "is_code_mixed"
    reduction_map = {"mean": [main_score]}
    prediction_type = "str"

    inference_model: InferenceEngine = None

    _requirements_list: List[str] = ["transformers", "torch"]

    def prepare(self):
        if IsCodeMixed.inference_model is None:
            IsCodeMixed.inference_model = HFPipelineBasedInferenceEngine(
                model_name="Nexusflow/Starling-LM-7B-beta",
                max_new_tokens=1,
                lazy_load=True,
            )
        # the processing steps for preparing the prompt (instruction, answer prefix etc.)
        # that we send to the generative model
        self.processor = SequentialOperator(
            steps=[
                "tasks.language_identification",
                "templates.language_identification.simple",
                "formats.models.starling",
            ]
        )

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict],
    ) -> dict:
        processed_data = self._prepare_instances_for_model(predictions)
        preds = IsCodeMixed.inference_model.infer(processed_data)

        # where the generated outputs begin with a number, the text gets a score of 1 (i.e., code-mixed)
        scores = [int(pred.isnumeric()) for pred in preds]
        return [{self.main_score: s} for s in scores]

    def _prepare_instances_for_model(self, texts: List[str]):
        stream = MultiStream(
            {
                "test": [{"text": text, "label": ""} for text in texts],
            }
        )
        processed_stream = self.processor.process(stream)
        return processed_stream.to_dataset()["test"]
