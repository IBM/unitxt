import re
import string
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import field
from statistics import mean
from typing import Any, Dict, Generator, List, Optional, Tuple

import evaluate
import numpy
import numpy as np
from scipy.stats import bootstrap

from .artifact import Artifact
from .dataclass import InternalField, OptionalField
from .logging_utils import get_logger
from .operator import (
    MultiStreamOperator,
    SingleStreamOperator,
    StreamingOperator,
    StreamInstanceOperator,
)
from .operators import CopyFields
from .random_utils import get_seed
from .stream import MultiStream, Stream
from .type_utils import isoftype

logger = get_logger()
# The default number of resamples used to estimate the confidence intervals
# global and instances metrics. Use None to disable confidence interval computation by default.
_N_RESAMPLES_DEFAULT_FOR_INSTANCE_METRICS = 1000
_N_RESAMPLES_DEFAULT_FOR_GLOBAL_METRICS = 100


def abstract_factory():
    return {}


def abstract_field():
    return field(default_factory=abstract_factory)


class UpdateStream(StreamInstanceOperator):
    update: dict

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance.update(self.update)
        return instance


# TODO: currently we have two classes with this name. metric.Metric and matrics.Metric...
class Metric(Artifact):
    @property
    @abstractmethod
    def main_score(self):
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
    def average_instance_scores(instances, field_name):
        """Calculate mean of a set of instance scores (given by field_name).

        Args:
            instances: The instances for which the confidence intervals are computed; should already have the relevant instance scores calculated.
            field_name: score field names to compute mean for.
        """
        return mean(
            [instance["score"]["instance"][field_name] for instance in instances]
        )

    def score_based_confidence_interval(
        self,
        instances: List[dict],
        score_names: List[str],
        aggregation_func=None,
        ci_score_prefix="",
    ):
        """Compute confidence intervals based on existing scores, already computed on the input instances.

        Unlike GlobalMetric, this is simply a function of the instance scores (possibly taking into account additional_inputs field),
         so they don't need to be recomputed after every bootstrap draw.

        Args:
            instances: The instances for which the confidence intervals are computed; should already have the relevant instance scores calculated.
            score_names: List of instance score field names to compute a confidence interval for.
            aggregation_func: A function with arguments instances, field_name; is applied on list of instances (which may include additional_inputs
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
            aggregation_func = self.average_instance_scores

        for score_name in score_names:
            # need to redefine the statistic function within the loop because score_name is a loop variable, to avoid ruff errors
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
        self, references, predictions, additional_inputs, score_name
    ):
        """Computed confidence intervals for a set of references and predictions."""
        random_gen = self.new_random_generator()

        def statistic(arr, axis):
            # arr is a 2d array where each row is a resampling, so we
            # iterate over the rows and compute the metric on each resampling
            def metric(sample_refs, sample_preds, sample_additional_inputs):
                try:
                    return self._compute(
                        references=sample_refs,
                        predictions=sample_preds,
                        additional_inputs=sample_additional_inputs,
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
                    sample_additional_inputs=[additional_inputs[i] for i in x],
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


class GlobalMetric(SingleStreamOperator, MetricWithConfidenceInterval):
    """A class for computing metrics that require joint calculations over all instances and are not just aggregation of scores of individuals instances.

    For example, macro_F1 requires
    calculation requires calculation of recall and precision per class, so all instances of the class
    need to be considered.  Accuracy, on the other hand, is just an average of the accuracy of all the instances.
    """

    n_resamples = _N_RESAMPLES_DEFAULT_FOR_GLOBAL_METRICS

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        references = []
        predictions = []
        additional_inputs = []
        global_score = {}

        instances = []

        for instance in stream:
            if "score" not in instance:
                instance["score"] = {"global": global_score, "instance": {}}
            else:
                global_score = instance["score"]["global"]

            instance_references, instance_prediction = (
                instance["references"],
                instance["prediction"],
            )
            references.append(instance_references)
            predictions.append(instance_prediction)
            instances.append(instance)

            instance_additional_inputs = (
                instance["additional_inputs"] if "additional_inputs" in instance else {}
            )
            additional_inputs.append(instance_additional_inputs)
            try:
                instance_score = self._compute(
                    [instance_references],
                    [instance_prediction],
                    [instance_additional_inputs],
                )
            except:
                instance_score = {"score": None, "score_name": self.main_score}

                if isinstance(self.main_score, str):
                    instance_score[self.main_score] = None

            instance["score"]["instance"].update(instance_score)

        result = self._compute(references, predictions, additional_inputs)

        global_score.update(result)

        score_name = global_score["score_name"]
        confidence_interval = self.compute_global_confidence_intervals(
            references, predictions, additional_inputs, score_name
        )
        global_score.update(confidence_interval)

        for instance in instances:
            instance["score"]["global"] = global_score
            yield instance

    def _compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        additional_inputs: List[Any],
    ) -> dict:
        result = self.compute(references, predictions, additional_inputs)
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result

    @abstractmethod
    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        additional_inputs: List[Any],
    ) -> dict:
        pass


class BulkInstanceMetric(SingleStreamOperator, MetricWithConfidenceInterval):
    n_resamples = _N_RESAMPLES_DEFAULT_FOR_INSTANCE_METRICS
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
                    (instance["references"], instance["prediction"])
                    for instance in stream
                ]
            ),
        )

        additional_inputs = [
            instance["additional_inputs"] if "additional_inputs" in instance else {}
            for instance in stream
        ]

        # compute the metric over all refs and preds
        instance_scores = self.compute(
            references=references,
            predictions=predictions,
            additional_inputs=additional_inputs,
        )

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
                for field_name in fields:
                    global_score[field_name] = mean(
                        [
                            instance["score"]["instance"][field_name]
                            for instance in instances
                        ]
                    )
                    if field_name == self.main_score:
                        global_score["score"] = global_score[field_name]
                        global_score["score_name"] = self.main_score

                ci_fields = (
                    list(set(self.ci_scores))
                    if self.ci_scores is not None
                    else [self.main_score]
                )
                confidence_interval = self.score_based_confidence_interval(
                    instances=instances, score_names=ci_fields
                )
                global_score.update(confidence_interval)

        for instance in instances:
            yield instance

    @abstractmethod
    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        additional_inputs: List[Dict],
    ) -> List[Dict[str, Any]]:
        pass


class InstanceMetric(SingleStreamOperator, MetricWithConfidenceInterval):
    n_resamples = _N_RESAMPLES_DEFAULT_FOR_INSTANCE_METRICS

    implemented_reductions: List[str] = field(
        default_factory=lambda: ["mean", "group_mean"]
    )

    # InstanceMetric currently allows two reductions, 'mean', which calculates the mean of instance scores,'
    # and 'group_mean', which first applies an aggregation function specified in the reduction_map
    # to instance scores grouped by the field grouping_field (which must not be None), and returns the mean
    # of the group scores; if grouping_field is None, grouping is disabled.
    # see _validate_group_mean_reduction for an example and proper formatting of the reduction_map
    grouping_field: str = None

    @property
    @abstractmethod
    def reduction_map(self) -> dict:
        pass

    def _validate_group_mean_reduction(self):
        """Ensure that group_mean reduction_map is properly formatted.

        Example: Apply the variance (np.var) to group Accuracy instance scores.  This class would be specified as follows:

        class GroupVarianceAccuracy(Accuracy):
            grouping_field = 'group_id'
            reduction_map = {'group_mean': {'agg_func': ['variance', np.var]}}

        reduction_map must be a dict with
        - an 'agg_func' field with value being a 2-element list where
            - 1st element is a string name of the aggregation function (used in naming the CI report)
            - 2nd element is the callable aggregation function
        - Optional: 'score_fields' key with list value containing the string names of fields to apply the aggregation to
            - if not present, the parent class main_score is used.
        """
        if not self.grouping_field:
            raise ValueError(
                "self.grouping_field is None, . "
                "This field is required for group based metric computation."
            )

        assert (
            "group_mean" in self.reduction_map
        ), "reduction_map must have a `group_mean' key"
        fields = self.reduction_map["group_mean"]
        # for group_mean, expects a dict
        assert isinstance(fields, dict)
        assert (
            "agg_func" in fields
        ), "fields should have a key 'agg_func' consisting of a 2-element list of a function name and function definition"
        assert isinstance(
            fields["agg_func"], list
        ), "fields['agg_func'] should be a list"
        assert (
            len(fields["agg_func"]) == 2
        ), "fields['agg_func'] should be a two-element list"
        assert isinstance(
            fields["agg_func"][0], str
        ), "first item in fields['agg_func'] should be a string name of a function"
        assert callable(
            fields["agg_func"][1]
        ), "second item in fields['agg_func'] should be a callable function"
        if "score_fields" in fields:
            assert isinstance(fields["score_fields"], list)

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        instances, global_score = self.compute_instance_scores(stream)

        for reduction_type, reduction_params in self.reduction_map.items():
            assert (
                reduction_type in self.implemented_reductions
            ), f"Reduction {reduction_type} is not implemented, use one of {self.implemented_reductions}"

            field_name_full_prefix = ""
            if reduction_type == "mean":
                aggregation_func = self.average_instance_scores
                reduction_fields = list(set(reduction_params))
            elif reduction_type == "group_mean":
                self._validate_group_mean_reduction()
                reduction_fields = (
                    [self.main_score]
                    if "score_fields" not in reduction_params
                    else list(set(reduction_params["score_fields"]))
                )
                aggregation_function_name = str(reduction_params["agg_func"][0])
                field_name_full_prefix = "group_" + aggregation_function_name + "_"

                def aggregation_func(
                    instances,
                    field_name,
                    group_aggregation_func=reduction_params["agg_func"][1],
                ):
                    return self.aggregate_instance_scores_by_group(
                        instances, field_name, group_aggregation_func
                    )
            else:
                raise ValueError(
                    f"Reduction {reduction_type} is not supported, please specify a valid reduction method in reduction_map {self.reduction_map}."
                )

            # calculate global scores for each reduction field
            for field_name in reduction_fields:
                field_name_full = field_name_full_prefix + field_name
                global_score[field_name_full] = aggregation_func(instances, field_name)
                if field_name == self.main_score:
                    global_score["score"] = global_score[field_name_full]
                    global_score["score_name"] = field_name_full

            # need to specify which fields should have CIs calculated for them through ci_scores
            # (will not automatically calculate CIs for fields in reduction map)
            if self.ci_scores is not None:
                confidence_interval = self.score_based_confidence_interval(
                    instances=instances,
                    aggregation_func=aggregation_func,
                    score_names=list(set(self.ci_scores)),
                    ci_score_prefix=field_name_full_prefix,
                )
                global_score.update(confidence_interval)

        yield from instances

    def compute_instance_scores(
        self, stream: Stream, stream_name: Optional[str] = None
    ):
        global_score = {}
        instances = []

        for instance in stream:
            refs, pred = instance["references"], instance["prediction"]
            additional_inputs = (
                instance["additional_inputs"] if "additional_inputs" in instance else {}
            )

            instance_score = self.compute(
                references=refs, prediction=pred, additional_inputs=additional_inputs
            )
            instance_score["score"] = instance_score[self.main_score]
            instance_score["score_name"] = self.main_score
            if "score" not in instance:
                instance["score"] = {"global": global_score, "instance": {}}
            else:
                global_score = instance["score"]["global"]

            instance["score"]["instance"].update(instance_score)

            instances.append(instance)

        return instances, global_score

    def aggregate_instance_scores_by_group(
        self, instances, field_name, group_aggregation_func
    ):
        from collections import defaultdict

        group_to_instance_scores = defaultdict(list)
        for instance in instances:
            additional_inputs = instance["additional_inputs"]
            if self.grouping_field not in additional_inputs:
                raise ValueError(
                    f"Missing '{self.grouping_field}' from instance {instance}. "
                    f"This field is required for group based metric computation."
                )
            group_key = additional_inputs[
                self.grouping_field
            ]  # do we need to convert to str?
            group_to_instance_scores[group_key].append(
                instance["score"]["instance"][field_name]
            )

        group_total_scores = [
            group_aggregation_func(scores)
            for scores in group_to_instance_scores.values()
        ]
        import warnings

        with warnings.catch_warnings():
            # final mean should be mean of group_total_score, ignoring NaN, hence nanmean
            # but if the group function values is NaN for ALL groups, nanmean throws a
            # RuntimeWarning that it is calculating the mean of an empty slice (with no non-Nans)
            # this is the desired behavior, but we want to avoid the warning here
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(group_total_scores)

    @abstractmethod
    def compute(
        self, references: List[Any], prediction: Any, additional_inputs: Dict
    ) -> dict:
        pass


class Squad(GlobalMetric):
    _metric = None
    main_score = "f1"
    metric = "squad"

    def prepare(self):
        super().prepare()
        self._metric = evaluate.load(self.metric)

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        additional_inputs: List[Dict],
    ) -> dict:
        ids = [str(uuid.uuid4()).replace("-", "") for _ in range(len(predictions))]
        formatted_predictions = [
            {"prediction_text": prediction, "id": ids[i]}
            for i, prediction in enumerate(predictions)
        ]
        formatted_references = [
            {"answers": {"answer_start": [-1], "text": reference}, "id": ids[i]}
            for i, reference in enumerate(references)
        ]

        return self._metric.compute(
            predictions=formatted_predictions,
            references=formatted_references,
        )


class Accuracy(InstanceMetric):
    reduction_map = {"mean": ["accuracy"]}
    main_score = "accuracy"
    ci_scores = ["accuracy"]

    def compute(
        self, references: List[Any], prediction: Any, additional_inputs: List[Dict]
    ) -> dict:
        result = {
            self.main_score: float(
                str(prediction) in [str(reference) for reference in references]
            )
        }
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result


class StringContainment(InstanceMetric):
    reduction_map = {"mean": ["string_containment"]}
    main_score = "string_containment"
    ci_scores = ["string_containment"]

    def compute(
        self, references: List[Any], prediction: Any, additional_inputs: List[Dict]
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
        additional_inputs: List[Dict],
    ) -> dict:
        passed_additional_inputs = {}
        for additional_input_field in self.hf_additional_input_fields:
            assert (
                additional_input_field in additional_inputs[0]
            ), f"'{additional_input_field}' field required by {__class__.__name__} is not in passed in additional inputs: {additional_inputs[0]}"
            passed_additional_inputs[additional_input_field] = [
                additional_input[additional_input_field]
                for additional_input in additional_inputs
            ]
        for additional_input_field in self.hf_additional_input_fields_pass_one_value:
            assert (
                additional_input_field in additional_inputs[0]
            ), f"'{additional_input_field}' field required by {__class__.__name__} is not in passed in additional inputs: {additional_inputs[0]}"

            values = {
                additional_input[additional_input_field]
                for additional_input in additional_inputs
            }
            assert (
                len(values) == 1
            ), f"Values of '{additional_input_field}' field required by {__class__.__name__}  should all be the same, but have multiple values {values}"

            passed_additional_inputs[additional_input_field] = next(iter(values))

        # add check that all required fields in self.metrics are in passed_additional_inputs       print(passed_additional_inputs)
        result = self.metric.compute(
            predictions=predictions,
            references=references,
            **passed_additional_inputs,
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
        self.metric = evaluate.load(self.hf_metric_name)

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        additional_inputs: List[Any],
    ) -> List[Dict[str, Any]]:
        passed_additional_inputs = {}
        for additional_input_field in self.hf_additional_input_fields:
            assert (
                additional_input_field in additional_inputs[0]
            ), f"'{additional_input_field}' field required by {__class__.__name__} is not in passed in additional inputs: {additional_inputs[0]}"
            passed_additional_inputs[additional_input_field] = [
                additional_input[additional_input_field]
                for additional_input in additional_inputs
            ]
        # add check that all required fields in self.metrics are in passed_additional_inputs

        scores = self.metric.compute(
            predictions=predictions,
            references=references,
            **passed_additional_inputs,
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

    def prepare(self):
        super().prepare()
        self._metric = evaluate.load(self.metric)

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
        additional_inputs: List[Dict],
    ) -> dict:
        assert all(
            len(reference) == 1 for reference in references
        ), "Only a single reference per prediction is allowed in F1 metric"
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


class F1Weighted(F1):
    main_score = "f1_weighted"
    average = "weighted"


class F1MultiLabel(GlobalMetric):
    _metric = None
    main_score = "f1_macro"
    average = None  # Report per class then aggregate by mean
    classes_to_ignore = ["none"]
    metric = "f1"

    def prepare(self):
        super().prepare()
        self._metric = evaluate.load(self.metric, "multilabel")

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
        additional_inputs: List[Dict],
    ) -> dict:
        self.str_to_id = {}
        self.id_to_str = {}

        self._validate_references_and_prediction(references, predictions)
        references = [reference[0] for reference in references]

        labels = [
            lbl
            for lbl in {label for reference in references for label in reference}
            if lbl not in self.classes_to_ignore
        ]
        # if no classes are left then F1 is not defined
        # (e.g. only "none" in references)
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
            from statistics import mean

            assert (
                len(result[self.metric]) == len(labels)
            ), f"F1 result ({result[self.metric]}) has more entries than labels ({labels})"
            final_result = {self.main_score: mean(result[self.metric])}
            for i, label in enumerate(labels):
                final_result[self.metric + "_" + label] = result[self.metric][i]
        else:
            final_result = {self.main_score: result[self.metric]}
        return final_result

    def _validate_references_and_prediction(self, references, predictions):
        for reference in references:
            if not len(reference) == 1:
                raise ValueError(
                    f"Only a single reference per prediction is allowed in F1 multi label metric. Received reference: {reference}"
                )
            if not isoftype(reference[0], List[str]):
                raise ValueError(
                    f"Each reference is expected to be a list of strings in F1 multi label metric. Received reference: '{reference[0]}'"
                )

        for prediction in predictions:
            if not isoftype(prediction, List[str]):
                raise ValueError(
                    f"Each prediction is expected to be a list of strings in F1 multi label metric. Received prediction: '{prediction}'"
                )


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

    use_aggregator: bool = True
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    sent_split_newline: bool = True

    def prepare(self):
        super().prepare()

        self.hf_compute_args.update(
            {"use_aggregator": self.use_aggregator, "rouge_types": self.rouge_types}
        )

        import nltk

        nltk.download("punkt")
        self.sent_tokenize = nltk.sent_tokenize

    def compute(self, references, predictions, additional_inputs: List[Dict]):
        if self.sent_split_newline:
            predictions = [
                "\n".join(self.sent_tokenize(prediction.strip()))
                for prediction in predictions
            ]
            references = [
                ["\n".join(self.sent_tokenize(r.strip())) for r in reference]
                for reference in references
            ]
        return super().compute(references, predictions, additional_inputs)


# Computes char edit distance, ignoring whitespace
class CharEditDistanceAccuracy(InstanceMetric):
    reduction_map = {"mean": ["char_edit_dist_accuracy"]}
    main_score = "char_edit_dist_accuracy"
    ci_scores = ["char_edit_dist_accuracy"]

    def prepare(self):
        super().prepare()
        import editdistance

        self.eval = editdistance.eval

    def compute(
        self, references, prediction: str, additional_inputs: List[Dict]
    ) -> dict:
        assert (
            len(references) == 1
        ), f"Expected only one reference , but received: {references}"

        formatted_prediction = "".join(prediction.split())
        formatted_reference = "".join(references[0].split())
        max_length = max(len(formatted_reference), len(formatted_prediction))
        if max_length == 0:
            return {"char_edit_dist_accuracy": 0.0}
        edit_dist = self.eval(formatted_reference, formatted_prediction)
        return {"char_edit_dist_accuracy": (1 - edit_dist / max_length)}


class Wer(HuggingfaceMetric):
    hf_metric_name = "wer"
    main_score = "wer"

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        additional_inputs: List[Dict],
    ) -> dict:
        assert all(
            len(reference) == 1 for reference in references
        ), "Only single reference per prediction is allowed in wer metric"
        formatted_references = [reference[0] for reference in references]
        result = self.metric.compute(
            predictions=predictions, references=formatted_references
        )
        return {self.main_score: result}


class MatthewsCorrelation(HuggingfaceMetric):
    hf_metric_name = "matthews_correlation"
    main_score = "matthews_correlation"
    str_to_id: dict = InternalField(default_factory=dict)

    def get_str_id(self, str):
        if str not in self.str_to_id:
            id = len(self.str_to_id)
            self.str_to_id[str] = id
        return self.str_to_id[str]

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        additional_inputs: List[Dict],
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


class CustomF1(GlobalMetric):
    main_score = "f1_micro"
    groups = None
    zero_division = 0.0

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

    def get_groups(self, elements, additional_inputs):
        groups = set()
        for sublist, additional_input in zip(elements, additional_inputs):
            for e in sublist:
                if self.should_ignore_element(e, additional_input):
                    continue
                groups.add(self.get_element_group(e, additional_input))
        return groups

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        additional_inputs: List[Dict],
    ) -> dict:
        # in case reference are List[List[List[Any]]] and predictions are List[List[Any]]:
        if (
            isinstance(references[0], list)
            and len(references[0]) > 0
            and isinstance(references[0][0], list)
        ):
            references = [element[0] for element in references]

        assert len(references) == len(predictions), (
            f"references size ({len(references)})"
            f" doesn't mach predictions sise ({len(references)})."
        )

        if self.groups is None:
            groups = self.get_groups(references, additional_inputs)
        else:
            groups = self.groups
        groups_statistics = {}
        for references_batch, predictions_batch, additional_input in zip(
            references, predictions, additional_inputs
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

        amount_of_predictions = pd_total
        if amount_of_predictions == 0:
            result["in_classes_support"] = 1.0
        else:
            result["in_classes_support"] = (
                1.0 - num_of_unknown_class_predictions / amount_of_predictions
            )
        result["f1_micro"] = self.f1(pn_total, pd_total, rn_total, rd_total)
        result["recall_micro"] = self.recall(pn_total, pd_total, rn_total, rd_total)
        result["precision_micro"] = self.precision(
            pn_total, pd_total, rn_total, rd_total
        )
        return result


class NER(CustomF1):
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

    def compute(
        self, references: List[Any], prediction: Any, additional_inputs: List[Dict]
    ) -> dict:
        results = [
            self._compute_single_ref(reference, prediction) for reference in references
        ]
        return {
            measure: max(r[i] for r in results)
            for i, measure in enumerate(["precision", "recall", "f1"])
        }

    def _compute_single_ref(
        self, reference: Any, prediction: Any
    ) -> Tuple[float, float, float]:
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
    hf_metric_name = "bertscore"
    main_score = "f1"
    reduction_map = {"mean": ["f1", "precision", "recall"]}
    hf_metric_fields = ["f1", "precision", "recall"]
    ci_scores = ["f1", "precision", "recall"]
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
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util as sbert_util

        self.model = SentenceTransformer(self.model_name)
        self.util = sbert_util

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        additional_inputs: List[Dict],
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
        preds_emb = self.model.encode(predictions)
        refs_emb = self.model.encode(
            [ref for ref_group in references for ref in ref_group]
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

    def prepare(self):
        super().prepare()
        from transformers import pipeline

        self.pipe = pipeline("text-classification", model=self.model_name)

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        additional_inputs: List[Dict],
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


class Perplexity(BulkInstanceMetric):
    """Computes the likelihood of generating text Y after text X - P(Y|X)."""

    main_score = "perplexity"
    reduction_map = {"mean": ["perplexity"]}

    perplexity_prompt: str

    batch_size: int = 32
    model_name: str

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        additional_inputs: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Computes the likelihood of generating text Y after text X - P(Y|X).

        :param references: the list of Y texts as a list of singletons.
        :param predictions: the list of X texts as a plain list of strings

        :return: the likelihood of generating text Y_i after text X_i = P(Y_i|X_i) for every i.
        """
        sources = []
        targets = []
        for prediction, instance_references in zip(predictions, references):
            for instance_reference in instance_references:
                sources.append(f"{self.perplexity_prompt} {prediction}")
                targets.append(instance_reference)

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        lm = (
            self.EncoderDecoderLM(model_name=self.model_name)
            if config.is_encoder_decoder is True
            else self.DecoderOnlyLM(model_name=self.model_name)
        )

        # compute P(Q|P) and store in queue
        scores = lm.compute_lm(
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
            instance_scores[self.main_score] = mean(instance_scores_list)

            instance_scores[self.main_score] = mean(instance_scores_list)
            all_instances_scores.append(instance_scores)

        return all_instances_scores

    class AbstractLM(ABC):
        def __init__(self, model_name):
            import torch
            from transformers import AutoTokenizer

            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = self.model_class().from_pretrained(self.model_name)
            self.is_cuda = torch.cuda.is_available()

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
                            batch_target, padding=True, return_tensors="pt"
                        )

                        # compute the logits
                        logits, labels = self.compute_batch(
                            tokens_source, tokens_target
                        )

                        # the model returns mean over all batch. We run the CE again without reduction
                        # and extarct the mean for each document
                        loss_fct = torch.nn.CrossEntropyLoss(
                            ignore_index=-100, reduction="none"
                        )
                        loss = loss_fct(
                            logits.view(-1, logits.size(-1)), labels.view(-1)
                        )
                        loss = loss.view(len(batch_source), -1)

                        # for each document, do mean only over the non zero values (sum(labels>0))
                        batch_loss = torch.sum(loss, dim=1) / torch.sum(
                            labels > 0, dim=1
                        )

                        # append the batch scores to the list of all scores
                        scores.append(batch_loss)

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
            tokens_docs_ids = tokens_source["input_ids"]
            attention = tokens_source["attention_mask"]
            labels = tokens_target["input_ids"]

            if self.is_cuda:
                tokens_docs_ids, attention, labels = (
                    tokens_docs_ids.cuda(),
                    attention.cuda(),
                    labels.cuda(),
                )

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

            if self.is_cuda:
                tokens, attention, labels = (
                    tokens.cuda(),
                    attention.cuda(),
                    labels.cuda(),
                )

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

    def prepare(self):
        from sklearn.metrics import ndcg_score

        super().prepare()
        self.eval = ndcg_score

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        additional_inputs: List[Any],
    ) -> dict:
        from collections import defaultdict
        from statistics import mean

        query_to_predictions_and_references = defaultdict(lambda: [[], []])
        for reference, pred, inputs_dict in zip(
            references, predictions, additional_inputs
        ):
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
    def compute(
        self, references: List[Any], prediction: Any, additional_inputs: Dict
    ) -> dict:
        # digest input
        pred_ids: List[Any] = prediction
        ref_ids: List[Any] = list(dict.fromkeys(references))

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
    def get_element_group(self, element, additional_input):
        return additional_input["keypoint"]

    def get_element_representation(self, element, additional_input):
        return additional_input["keypoint"]

    def should_ignore_element(self, element, additional_input):
        return element == "none"


# define metrics that return means of an aggregation function applied across levels of a grouping variable
def performance_drop_rate(instance_scores: List):
    """Percentage change of mean performance on test elements relative to that on a baseline.

    from https://arxiv.org/pdf/2306.04528.pdf.

    Args:
        instance_scores: a list of scores on instances.  Assume the first element is the original, the others are test set

    Returns:
        numeric PDR metric.
        If only one element (no test set) or the first is 0 (percentage change is undefined) return NaN
        otherwise, calculate PDR

    """
    assert isinstance(instance_scores, list)
    return (
        np.nan
        if (len(instance_scores) < 2 or instance_scores[0] == 0)
        else 1 - mean(instance_scores[1:]) / instance_scores[0]
    )


def normalized_cohens_h(instance_scores: List):
    """Cohen's h between two proportions.

    Allows for change-type metric when the baseline is 0 (percentage change, and thus PDR, is undefined)
    https://en.wikipedia.org/wiki/Cohen%27s_h

    Cohen's h effect size metric between two proportions p2 and p1 is 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1))).
    h in -pi, pi, with +/- representing the largest increase/decrease (p1=0, p2=1), or (p1=1, p2=0).
    h=0 is no change. Unlike percentage change, h is defined if the baseline (p1) is 0.
    Assumes the scores are in [0,1], either continuous or binary.
    For scores in a list, the first element is treated as the baseline p1, and the mean of the others
    as p2, and evaluates p2 change relative to p1.  It is thus undefined if the list is of length < 2.
    We rescale this to [-1,1] from [-pi,pi] for clarity, where +- 1 are the most extreme changes, and 0 is no change

    Args:
        instance_scores: a list of scores on instances.  Assume the first element is the original, the others are test set

    Returns:
        float score between -1 and 1
    """
    assert isinstance(instance_scores, list)
    assert all(
        0 <= score <= 1 for score in instance_scores
    ), "all scores must be in [0,1]"

    if len(instance_scores) < 2:
        # needs at least 2 elements
        return np.nan
    # assumes first element is the baseline proportion
    baseline_p = instance_scores[0]
    new_p = np.nanmean(instance_scores[1:])
    h = 2 * (np.arcsin(np.sqrt(new_p)) - np.arcsin(np.sqrt(baseline_p)))
    return np.clip(a=h / np.pi, a_min=-1, a_max=1)


class GroupMeanAccuracy(Accuracy):
    grouping_field = "group_id"
    reduction_map = {"group_mean": {"agg_func": ["mean", mean]}}


class GroupPDRAccuracy(Accuracy):
    grouping_field = "group_id"
    reduction_map = {"group_mean": {"agg_func": ["pdr", performance_drop_rate]}}


class GroupMeanStringContainment(StringContainment):
    grouping_field = "group_id"
    reduction_map = {"group_mean": {"agg_func": ["mean", np.nanmean]}}


class GroupPDRStringContainment(StringContainment):
    grouping_field = "group_id"
    reduction_map = {"group_mean": {"agg_func": ["pdr", performance_drop_rate]}}


class GroupMeanTokenOverlap(TokenOverlap):
    grouping_field = "group_id"
    reduction_map = {
        "group_mean": {
            "agg_func": ["mean", np.nanmean],
            "score_fields": ["f1", "precision", "recall"],
        }
    }


class GroupNormCohensHAccuracy(Accuracy):
    grouping_field = "group_id"
    reduction_map = {"group_mean": {"agg_func": ["norm_cohens_h", normalized_cohens_h]}}


class GroupNormCohensHStringContainment(StringContainment):
    grouping_field = "group_id"
    reduction_map = {"group_mean": {"agg_func": ["norm_cohens_h", normalized_cohens_h]}}
