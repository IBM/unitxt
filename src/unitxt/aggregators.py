import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy
import numpy as np
from scipy.stats import bootstrap

from .artifact import Artifact
from .dict_utils import dict_get
from .operators import FilterByCondition
from .random_utils import get_seed


class Aggregator(Artifact):
    """Aggregates over a list of instances, updating the score/global field of each of them by a dictionary of scores."""

    score_names: List[str] = None

    # currently, for backward compatibility, non trivial only for grouper_aggregator
    aggregator_based_score_prefix = ""

    # computes global score, and accordingly updates instance["score"]["global"] of each instance
    @abstractmethod
    def aggregate(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass

    # prefix and add "score" and "score_name"
    def prefix_global_scores(self, gs: Dict[str, Any]) -> Dict[str, Any]:
        full_prefix = self.aggregator_based_score_prefix + self.score_prefix
        new_gs = {}
        new_gs["score_name"] = full_prefix + self.main_score
        new_gs["score"] = gs[self.main_score]
        if self.main_score + "_ci_low" in gs:
            new_gs["score_ci_low"] = gs[self.main_score + "_ci_low"]
        if self.main_score + "_ci_high" in gs:
            new_gs["score_ci_high"] = gs[self.main_score + "_ci_high"]

        for score_name, score in gs.items():
            new_gs[full_prefix + score_name] = score

        return new_gs

    # trivial for all but grouperaggregator
    def instances_to_sample_from_and_sample_aggregator(
        self, instances: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Any]:
        return (instances, self)

    def set_metric_related_properties(
        self,
        is_serving_global_metric: bool,
        n_resamples: int,
        ci_scores: List[str],
        main_score: str,
        score_prefix: str = "",
        aggregating_function_name: str = "",
        confidence_level: float = 0.95,
    ):
        self.n_resamples = n_resamples
        self.ci_scores = ci_scores
        self.confidence_level = confidence_level
        self.main_score = main_score
        self.score_prefix = score_prefix
        self.aggregating_function_name = aggregating_function_name
        self.is_serving_global_metric = is_serving_global_metric

    def compute_ci(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        if (
            self.n_resamples is None
            or self.n_resamples <= 0
            or len(self.ci_scores) == 0
        ):
            return {}
        (
            instances_to_sample_from,
            aggregator,
        ) = self.instances_to_sample_from_and_sample_aggregator(instances)
        confidence_interval_calculator = ConfidenceIntervalCalculator(
            n_resamples=self.n_resamples,
            ci_scores=self.ci_scores,
            confidence_level=self.confidence_level,
            instances_to_sample_from=instances_to_sample_from,
            aggregator=aggregator,
            is_serving_global_metric=self.is_serving_global_metric,
        )
        return confidence_interval_calculator.compute_ci()

    # update instance["score"]["global"] with the newly computed global score, for the
    # current metric computed.  global_score contains "score" and "score_name" fields that reflect
    # (the main_score of) the current metric.
    # A simple python-dictionary-update adds new fields to instance["score"]["global"], and also replaces the values
    # of its fields "score" and "score_name", to reflect the current metric, overwriting previous metrics' settings
    # of these fields (if any previous metric exists).
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
    # instance["score"]["global"] are consistent with the current metric: The current metric
    # is named instance["score"]["global"]["score_name"], its score shows in
    # field instance["score"]["global"]["score"], and it does not have ci_scores,
    # which is also reflected in the absence of fields "score_ci_low" and "score_ci_high" from instance["score"]["global"].
    # If ci IS computed for the current metric, global_score contains "score_ci_low" and "score_ci_high", and these overwrite
    # the ones existing in instance["score"]["global"] by a simple python-dictionary-update, and no need for any further fixeup.
    def update_and_adjust_global_score(
        self, instance: Dict[str, Any], global_score: dict
    ):
        instance["score"]["global"].update(global_score)
        for score_ci in ["score_ci_low", "score_ci_high"]:
            if score_ci in global_score:
                continue
            if score_ci in instance["score"]["global"]:
                instance["score"]["global"].pop(score_ci)

    def update_global_score(self, instances: List[Dict[str, Any]], gs: Dict[str, Any]):
        for instance in instances:
            if "global" not in instance["score"]:
                instance["score"]["global"] = {}
            self.update_and_adjust_global_score(instance, gs)

    def compute_global_score(self, instances: List[Dict[str, Any]]):
        gs = self.aggregate(instances)
        gs.update(self.compute_ci(instances))
        gs = self.prefix_global_scores(gs)
        self.update_global_score(instances=instances, gs=gs)


class MeanAggregator(Aggregator):
    def aggregate(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {}
        for score_name in self.score_names:
            result[score_name] = nan_mean(
                [instance["score"]["instance"][score_name] for instance in instances]
            )

        return result


class MaxAggregator(Aggregator):
    def aggregate(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {}
        for score_name in self.score_names:
            result[score_name] = nan_max(
                [instance["score"]["instance"][score_name] for instance in instances]
            )

        return result


class FilterAggregator(Aggregator):
    aggregator: Aggregator
    filter_by_condition: FilterByCondition

    def aggregate(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        filtered_instances = [
            instance
            for instance in instances
            if self.filter_by_condition._is_required(instance)
        ]

        self.aggregator.score_names = self.score_names
        return self.aggregator.aggregate(instances=filtered_instances)


class GrouperAggregator(Aggregator):
    """Splits the input instances into groups by the value found in them in field "split_to_groups_by_query".

    Then, aggregates over each group yielding a dictionary of global scores.
    These dictionaries are then sorted by the values associated with each
    group, by which it was split from the other groups, and "dressed" like instance scores, and these are
    averaged, to return the result.
    """

    split_to_groups_by_query: str
    one_group_aggregator: Aggregator
    # the following boolean flag specifies whether resampling for CI
    # should be done from the individual groups' scores (True), as if each group is represented by
    # one instance whose instance["score"]["instance"][score_name] is the group's global score for score_name,
    # Or from the whole stream (False), where each resample is then split to
    # groups, the score of which is then computed, and finally averaged with the other groups' scores, as done
    # here for the original whole stream.
    ci_samples_from_groups_scores: bool = False

    def split_to_group_by(
        self, instances: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        res = defaultdict(list)
        for instance in instances:
            val = dict_get(instance, self.split_to_groups_by_query)
            res[val].append(instance)
        return res

    def gen_instances_from_group_scores(
        self, instances: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        ms_dict = self.split_to_group_by(instances)
        self.one_group_aggregator.score_names = self.score_names
        groups_scores = []
        for group_name in sorted(ms_dict.keys()):
            group_gs = self.one_group_aggregator.aggregate(
                instances=ms_dict[group_name]
            )
            groups_scores.append({"score": {"instance": group_gs}})
        return groups_scores

    def aggregate(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        groups_scores = self.gen_instances_from_group_scores(instances)
        over_groups_aggregator = MeanAggregator(score_names=self.score_names)
        return over_groups_aggregator.aggregate(groups_scores)

    def prefix_global_scores(self, gs: Dict[str, Any]) -> Dict[str, Any]:
        # for backward compatibility, only for this aggregator we have this informative prefix, perhaps now
        # is time to allow for other aggregators too
        self.aggregator_based_score_prefix = (
            "fixed_group_" if self.ci_samples_from_groups_scores else "group_"
        )
        if len(self.aggregating_function_name) > 0:
            self.aggregator_based_score_prefix += self.aggregating_function_name + "_"
        return super().prefix_global_scores(gs=gs)

    def instances_to_sample_from_and_sample_aggregator(
        self, instances: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Aggregator]:
        if self.ci_samples_from_groups_scores:
            return (
                self.gen_instances_from_group_scores(instances),
                MeanAggregator(score_names=self.score_names),
            )
        return (instances, self)


class ControlComparisonAggregator(Aggregator):
    # Generate a score that compares the scores of two subsets of the input stream: subset 'control' and subset 'comparison'
    # The input stream (instances) may be one group (in case that split_to_groups_by is not None), or per the whole
    # stream (in case that split_to_groups_by is None).
    control_comparison_subsets: Dict[
        Literal["control", "comparison"], FilterByCondition
    ]
    control_comparison_floats_comparator: Callable[[List[float], List[float]], float]
    return_abs_value: bool = False

    def aggregate(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        subsets_dict = {
            side: [
                instance
                for instance in instances
                if self.control_comparison_subsets[side]._is_required(instance)
            ]
            for side in ["control", "comparison"]
        }
        dict_to_return = {}
        for score_name in self.score_names:
            control_floats = [
                instance["score"]["instance"][score_name]
                for instance in subsets_dict["control"]
            ]
            comparison_floats = [
                instance["score"]["instance"][score_name]
                for instance in subsets_dict["comparison"]
            ]
            val = self.control_comparison_floats_comparator(
                control_subset=control_floats, comparison_subset=comparison_floats
            )

            dict_to_return[score_name] = np.abs(val) if self.return_abs_value else val
        return dict_to_return


class ConfidenceIntervalCalculator(Artifact):
    """Generates samples from the input instances, then aggregates each sample, and returns the percentiles."""

    n_resamples: int = None
    ci_scores: List[str] = None

    confidence_level: float = 0.95
    instances_to_sample_from: List[Dict[str, Any]]
    aggregator: Aggregator  # to aggregate over each resample
    is_serving_global_metric: bool  # needed for a precheck of all scores are equal

    @staticmethod
    def new_random_generator():
        # The np.random.default_rng expects a 32-bit int, while hash(..) can return a 64-bit integer.
        # So use '& MAX_32BIT' to get a 32-bit seed.
        _max_32bit = 2**32 - 1
        return np.random.default_rng(hash(get_seed()) & _max_32bit)

    def _can_compute_confidence_intervals(
        self, instances: List[Dict[str, Any]]
    ) -> bool:
        return (
            self.n_resamples is not None and self.n_resamples > 1 and len(instances) > 1
        )

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

    def compute_ci(self) -> Dict[str, Any]:
        """Compute confidence intervals based on existing scores, if exist, or compute via arg metric."""
        if not self._can_compute_confidence_intervals(self.instances_to_sample_from):
            return {}

        result = {}

        for score_name in self.ci_scores:
            if not self.is_serving_global_metric and self._all_instance_scores_equal(
                self.instances_to_sample_from, score_name
            ):
                continue

            # generate resamples from instances_to_sample_from, then aggregate over each.
            # and extract the percentiles thereof

            # need to redefine the statistic function within the loop because score_name is a loop variable
            def statistic(arr, axis, score_name=score_name):
                # arr is a 2d array where each row is a resampling, so we
                # iterate over the rows and compute the metric on each resampling
                scores = numpy.apply_along_axis(
                    lambda resampled_instances: self.aggregator.aggregate(
                        instances=resampled_instances
                    )[score_name],
                    axis=axis,
                    arr=arr,
                )
                return self.resample_from_non_nan(scores)

            with warnings.catch_warnings():
                # Avoid RuntimeWarning in bootstrap computation. This happens on small datasets where
                # the value of the computed global metric is the same on all resamplings.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ci = bootstrap(
                    (self.instances_to_sample_from,),
                    statistic=statistic,
                    n_resamples=self.n_resamples,
                    confidence_level=self.confidence_level,
                    random_state=self.new_random_generator(),
                ).confidence_interval
            result[score_name + "_ci_low"] = ci.low
            result[score_name + "_ci_high"] = ci.high

        return result


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
