import warnings
from collections import defaultdict, namedtuple
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import gmean, norm, permutation_test, ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

COMMON_REPORT_FIELDS = [
    "model_names",
    "metric",
    "pvalues",
    "alternative",
    "permute",
    "corrected",
    "pvalue_is_signif",
    "sample_means",
    "num_obs",
    "pvalue_alpha",
]


def pval_trans(p):
    # from R emmeans module (emmeans:::.pval.tran)
    # transform a pvalue for visualization
    if isinstance(p, float):
        p = np.array([p])
    bds = (5e-5, 1)
    xx = np.clip(p, a_min=bds[0], a_max=bds[1])
    rtn = norm.cdf(x=np.log(xx), loc=-2.5, scale=3) / 0.7976716
    spec = np.logical_or(p < bds[0], p > bds[1])
    rtn[spec] = p[spec]
    return rtn


def pval_inv(x):
    # from R emmeans module (emmeans:::.pval.inv)
    # transform a value back to the p-value for visualization
    if isinstance(x, float):
        x = np.array([x])
    bds = (5e-5, 0.99995)
    pp = np.clip(x, a_min=bds[0], a_max=bds[1])
    rtn = np.exp(norm.ppf(q=0.7976716 * pp, loc=-2.5, scale=3))
    spec = np.logical_or(x < bds[0], x > bds[1])
    rtn[spec] = x[spec]
    return rtn


def spectral_palette(n):
    import seaborn as sns

    # return a list of tuples specifying n equally spaced colors in the Spectral colormap, used to code n levels
    n = int(n)
    assert n >= 1 and n <= 256
    return sns.color_palette(palette="Spectral", n_colors=n, as_cmap=True)(
        np.linspace(0, 1, n)
    )


def obs_idx_is_never_nan(arr_list):
    """Check if an index in a list of arrays is never NaN.

    Args:
        arr_list: a list of numpy arrays of the same length

    Returns:
        boolean array, where index is True if that index is never NaN in any array, otherwise False
    """
    return np.apply_along_axis(
        arr=np.vstack(arr_list), axis=0, func1d=lambda x: np.isnan(x).sum() == 0
    )


class PairedDifferenceTest:
    """Class to conduct test of statistical significance (from 0) in average differences between sample values for the same observation index.

    Can be used to measure if a pair of prediction models differ substantially in the metric scores they
    receive across observations.
    """

    # samples must be formatted as a namedtuple
    SAMPLE = namedtuple("Sample", "data model metric")
    # output of signif_pair_diff will be a namedtuple in one of these formats
    PERMUTE_REPORT = namedtuple("DiffTest", " ".join(COMMON_REPORT_FIELDS))
    # additional fields for effect sizes available only if not permute
    TTEST_REPORT = namedtuple(
        "DiffTest",
        " ".join([*COMMON_REPORT_FIELDS, *["effect_sizes", "effect_size_is_signif"]]),
    )
    # for heatmap
    HEATMAP_MATRICES = namedtuple(
        "HeatmapMatrices",
        "recoded_values_arr, combined_results_arr_with_nan, combined_results combined_results_arr",
    )
    # minimum number of observations to allow to use McNemar's test
    MIN_BINARY_CONTINGENCY_SAMPLES = 5

    def __init__(self, nmodels=None, alpha=0.05, model_names=None):
        """Initialize an object to be used to compare the same set of models across different datasets and metrics.

        Args:
            nmodels: the number of models (samples) to be compared pairwise, must be integer >= 2
            alpha: statistical false positive rate confidence to be used in evaluating significance
            model_names: optional list of names for the models compared, of length nmodels; if None, will be called 'model 1', 'model 2', etc.
        """
        # desired false positive rate, criterion for p-values
        self.alpha = float(np.clip(alpha, a_min=1e-10, a_max=0.5))
        self.binary_values = np.array([0, 1])

        if nmodels is None:
            if model_names is not None:
                assert (
                    isinstance(model_names, list) and len(model_names) >= 2
                ), "model_names must be a list of length >= 2"
                self.model_names = [str(nm) for nm in model_names]
                self.nmodels = len(self.model_names)
            else:
                raise ValueError(
                    "at least one of nmodels or model_names must be specified"
                )
        else:
            # nmodels is not None
            assert (
                isinstance(nmodels, int) and nmodels >= 2
            ), "nmodels must be an integer >= 2"
            self.nmodels = max(2, int(nmodels))
            if model_names is not None:
                assert (
                    len(model_names) == self.nmodels
                    and len(set(model_names)) == self.nmodels
                ), f"length of model_names (={len(model_names)}) must equal nmodels (={self.nmodels}) and consist of {self.nmodels} unique strings"
                self.model_names = [str(nm) for nm in model_names]
            else:
                # create default model names
                self.model_names = [f"model {ii}" for ii in range(1, self.nmodels + 1)]

    @staticmethod
    def _permute_diff_statistic(lx, ly):
        """Statistic to be used in permutation test.

        Calculates expected difference between pairs (= difference in expected values of the samples themselves).

        Args:
            lx: first sample
            ly: second sample
        """
        obs_not_nan_for_all_samples = obs_idx_is_never_nan([lx, ly])
        # select only the pairs for which both values are not NaN
        return np.mean(lx[obs_not_nan_for_all_samples]) - np.mean(
            ly[obs_not_nan_for_all_samples]
        )

    def format_as_samples(self, samples_list, metric="unknown"):
        """Format a list of samples as a list of namedtuples for this object (assumes come from the model_names here).

        Args:
            samples_list: list of equal-length np.arrays, each representing a model's values of a given metric
            metric: string name of metric (score function) that the samples are realizations of
        Returns:
            list of namedtuples of type SAMPLE
        """
        assert isinstance(samples_list, list)
        assert (
            len(samples_list) == self.nmodels
        ), f"samples_list (length {len(samples_list)} must equal self.nmodels ({self.nmodels}"
        assert (
            len({len(ss) for ss in samples_list}) == 1
        ), "all samples in samples_list must be of the same length"
        assert (
            len(samples_list[0]) >= 2
        ), "samples must all have at least 2 observations"

        return [
            self.SAMPLE(ss, mn, str(metric))
            for ss, mn in zip(samples_list, self.model_names)
        ]

    def _check_valid_samples(self, samples_list):
        """Check if samples_list is valid input for significance testing (come from same models as model_names, etc.).

        Args:
            samples_list: a list of SAMPLES namedtuples
        """
        assert isinstance(samples_list, list)
        assert all(
            isinstance(ss, self.SAMPLE) for ss in samples_list
        ), "all elements in samples_list must be SAMPLE namedtuples"

        assert (
            len(samples_list) == self.nmodels
        ), f"samples_list (length {len(samples_list)} must equal self.nmodels ({self.nmodels}"
        assert (
            len({len(ss.data) for ss in samples_list}) == 1
        ), "all samples' data field in samples_list must be of the same length"
        assert all(
            mn == sn
            for mn, sn in zip(self.model_names, [ss.model for ss in samples_list])
        ), "samples_list model names must be the same as self.model_names"
        assert (
            len({ss.metric for ss in samples_list}) == 1
        ), "all samples' metric field must be the same"

    def _check_binary(self, samples_list):
        """Check if samples are binary-valued.

        Args:
            samples_list: a list of 1-D numpy arrays

        Returns:
            boolean: True if all observed values, ignoring NaNs, are binary 0/1, otherwise False
        """
        self._check_valid_samples(samples_list)
        uv = np.unique(np.vstack([ss.data for ss in samples_list]))
        return np.all(np.isin(uv[~np.isnan(uv)], self.binary_values))

    def _can_use_mcnemar(self, samples_list, alternative="two-sided"):
        """Check if can use the McNemar test (accepts only 2 binary equal-length samples) which simplifies computation.

        Args:
            samples_list: a list of 1-D numpy arrays
            alternative: alternative hypothesis to be used (only two-sided is valid)

        Returns:
            boolean
        """
        is_binary = self._check_binary(samples_list)
        obs_not_nan_for_all_samples = obs_idx_is_never_nan(
            [ss.data for ss in samples_list]
        )
        # require at least 5 obsservations for which the values in all samples are not NaN
        return (
            is_binary
            and (len(samples_list) == 2)
            and (alternative == "two-sided")
            and obs_not_nan_for_all_samples.sum() >= self.MIN_BINARY_CONTINGENCY_SAMPLES
        )

    def _handle_binary_data_contingency(self, samples_list, continuity_correction=True):
        """Convert samples into 2x2 contingency table form, even if some value combinations are not observed.

        Args:
            samples_list: a list of 1-D numpy arrays
            continuity_correction: whether to perform continuity correction for cells with 0 value

        Returns:
            2x2 numpy cross-tabulation array
        """
        # make sure that a cross tabulation is done that is 2x2 in case some value combinations are missing
        cat_binary = pd.CategoricalDtype(categories=self.binary_values, ordered=False)
        df = pd.DataFrame(np.transpose(np.vstack([ss.data for ss in samples_list])))
        # drop any rows for which there is at least one NaN value
        df.dropna(axis=0, how="any", inplace=True)
        # encode as categorical binary with all values
        df = df.astype(cat_binary)

        assert len(df) >= self.MIN_BINARY_CONTINGENCY_SAMPLES
        # need to allow floats for continuity correction; use dropna=False in crosstab to avoid dropping missing combinations
        return pd.crosstab(df[0], df[1], dropna=False).to_numpy()

    def mcnemar_test(self, samples_list):
        """Perform McNemar test and return a p-value and effect size.

        Use approximation (exact=False) so that effect size can be calculated

        Args:
            samples_list: a list of 1-D numpy arrays (must be binary)

        Returns:
            float
        """
        # assertion in case is used outside of signif_pair_diff
        assert len(samples_list) == 2, "McNemar's test can use only two samples"
        tbl = self._handle_binary_data_contingency(samples_list)
        # Cohen's g effect size, https://rcompanion.org/handbook/H_05.html#_Toc507754342
        # < 0.05 = very small; (0.05, 0.15) = small; (0.15, 0.25) = medium, >= 0.25 is large
        # do continuity correction for the effect size, but not for the p-value
        b, c = max(tbl[0, 1], 0.5), max(tbl[1, 0], 0.5)
        cohens_g = max(b / (b + c), c / (b + c)) - 0.5
        return mcnemar(table=tbl, exact=True).pvalue, cohens_g

    def iterate_pairs(self, samples_list=None):
        """Iterate over ordered combinations of samples or their indices (if samples_list is None).

        Args:
            samples_list: a list of 1-D numpy arrays (must be binary)

        Returns:
            iterator
        """
        # return pairs of values (samples) that are compared, or the indices if samples_list=None
        # notice, the pair order is important if one-sided tests are used because it affects the hypothesis order
        return combinations(
            range(self.nmodels) if samples_list is None else samples_list, r=2
        )

    def signif_pair_diff(
        self,
        samples_list,
        alternative="two-sided",
        permute=False,
        corrected=True,
        random_state=None,
    ):
        """Conduct paired-observation estimation of difference in means between samples.

        Args:
            samples_list: a list of SAMPLES namedtuples
            alternative: alternative hypothesis to be used; one-sided (greater/less) should only be used if the arrays have been ordered
            permute: whether or not to use permutation test (requires some simulation)
            corrected: whether or not to apply a correction for control of false positive rate given the number of hypotheses (used when nmodels > 2)
            random_state: either an integer seed or None

        Returns:
            dict
        """
        self._check_valid_samples(samples_list)
        # because sample ordering is not fixed, do either greater (one-sided) or two-sided
        assert alternative in ("greater", "less", "two-sided")
        if alternative != "two-sided":
            flds = (
                ("greater", "descending")
                if alternative == "greater"
                else ("less", "ascending")
            )
            warnings.warn(
                "If 'alternative' is '{}', the samples are expected to be sorted in {} order of ANTICIPATED (not OBSERVED) mean value".format(
                    *flds
                ),
                stacklevel=1,
            )
        if self.nmodels == 2:
            # require more than 2 hypotheses (only if nmodels > 2) to be able to do a correction
            corrected = False

        # first see if can use McNemar's test, better for binary data; only works for two-sided hypotheses with two samples
        # validity check
        if self._can_use_mcnemar(samples_list, alternative):
            pvalue, eff_size = self.mcnemar_test(
                samples_list=[ss.data for ss in samples_list]
            )
            permute = False
            res = {
                "pvalues": np.array([pvalue]),
                "pvalue_is_signif": np.array([pvalue <= self.alpha]),
                "effect_sizes": np.array([eff_size]),
                "effect_size_is_signif": np.array([eff_size >= 0.25]),
            }

        else:
            # most other cases if have continuous variables or more than 2 models to compare
            # greater when ttest_rel(a,b) checks if a_i - b_i > 0
            # note, regardless of the significance, the value of the paired expectation E(a_i - b_i) will be the same as the
            # difference in means, i.e. E(a) - E(b).

            # pvalue is low (close to 0) if E(a_i - b_i) is significant (according to alternative)
            res = {}
            if permute:
                pvalues = np.array(
                    [
                        permutation_test(
                            data=vec,
                            statistic=self._permute_diff_statistic,
                            alternative=alternative,
                            permutation_type="samples",
                            vectorized=False,
                            random_state=random_state,
                        ).pvalue
                        for vec in self.iterate_pairs(
                            samples_list=[ss.data for ss in samples_list]
                        )
                    ]
                )
            else:
                obs_is_not_nan_for_both = [
                    obs_idx_is_never_nan([vec[0], vec[1]])
                    for vec in self.iterate_pairs(
                        samples_list=[ss.data for ss in samples_list]
                    )
                ]

                test_res = [
                    ttest_rel(
                        a=vec[0][include], b=vec[1][include], alternative=alternative
                    )
                    for vec, include in zip(
                        self.iterate_pairs(
                            samples_list=[ss.data for ss in samples_list]
                        ),
                        obs_is_not_nan_for_both,
                    )
                ]
                pvalues = np.array([vv.pvalue for vv in test_res])
                # effect size is the statistic / sqrt(n), where we include only non NaNs; equivalent of Cohen's d statistic
                # see e.g. https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Paired_T-Tests_using_Effect_Size.pdf
                # permutation test doesn't have effect sizes
                # this affects only the statistic, not the reported num_obs in the report
                res["effect_sizes"] = np.array(
                    [
                        tr.statistic / np.sqrt(include.sum())
                        for tr, include in zip(test_res, obs_is_not_nan_for_both)
                    ]
                )

                if alternative == "two-sided":
                    # magnitude in either direction
                    res["effect_size_is_signif"] = np.abs(res["effect_sizes"]) >= 0.8
                else:
                    # has to be in the direction specified by the alternative
                    res["effect_size_is_signif"] = (
                        res["effect_sizes"] >= 0.8
                        if alternative == "greater"
                        else res["effect_sizes"] <= -0.8
                    )

            if corrected:
                mult_corr = multipletests(pvals=pvalues, alpha=self.alpha)
                res.update({"pvalues": mult_corr[1], "pvalue_is_signif": mult_corr[0]})
            else:
                res.update(
                    {"pvalues": pvalues, "pvalue_is_signif": pvalues <= self.alpha}
                )

        res.update(
            {
                "corrected": corrected,
                "alternative": alternative,
                "permute": permute,
                "sample_means": np.array([np.nanmean(ss.data) for ss in samples_list]),
                "metric": samples_list[0].metric,
                "model_names": self.model_names,
                "num_obs": len(samples_list[0].data),
                "pvalue_alpha": self.alpha,
            }
        )

        return (
            self.PERMUTE_REPORT(*[res[kk] for kk in self.PERMUTE_REPORT._fields])
            if permute
            else self.TTEST_REPORT(*[res[kk] for kk in self.TTEST_REPORT._fields])
        )

    def _check_valid_signif_report(self, test_res):
        """Test if an instance of results from signif_pair_diff is a valid comparison to an object of class PairedDifferenceTest.

        Args:
            test_res: a namedtuple outputted from signif_pair_diff, from the same item
        """
        try:
            assert isinstance(test_res, self.PERMUTE_REPORT) or isinstance(
                test_res, self.TTEST_REPORT
            )
            assert self.nmodels == len(
                test_res.model_names
            ), "must have the same number of models"
            assert all(
                ii == jj for ii, jj in zip(self.model_names, test_res.model_names)
            ), "model_names lists must match"
            assert (
                self.alpha == test_res.pvalue_alpha
            ), f"significance tests must have used alpha={self.alpha}"

        except ValueError as e:
            raise ValueError(
                "test_res is incompatible with this PairedDifferenceTest object"
            ) from e

    def pvalue_lineplot(self, test_res):
        """A lineplot that shows pairs of compared models and arranges them along the x-axis by p-value.

        Inspired by https://cran.r-project.org/web/packages/emmeans/vignettes/comparisons.html
        The graph plot is recommended because the lineplot can result in over-plotting and a busy image that is difficult to read

        Args:
            test_res: a namedtuple outputted from signif_pair_diff, from the same item

        """
        self._check_valid_signif_report(test_res)
        pal = spectral_palette(n=self.nmodels)

        level_order = np.argsort(test_res.sample_means).tolist()
        level_order = [level_order.index(ii) for ii in range(self.nmodels)]

        # pal now accepts the original sample index, but now ordered by the value of sample mean
        pal = [tuple(pal[ii]) for ii in level_order]

        tick_vals = np.unique(
            np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.5, 0.1, 0.2, 0.5, 1.0, self.alpha])
        )
        # make sure the axis range contains all pvalues including the alpha
        pval_range = np.array(
            [
                min(test_res.pvalues.min(), self.alpha),
                max(test_res.pvalues.max(), self.alpha),
            ]
        )
        tick_vals = np.array(
            [vv for vv in tick_vals if vv >= pval_range[0] and vv <= pval_range[1]]
        )

        # pad the axes to have 5% empty space on each end
        range_pad = 0.05
        xaxis_range = pval_trans(pval_range)
        yaxis_range = np.array(
            [test_res.sample_means.min(), test_res.sample_means.max()]
        )
        yaxis_range = tuple(
            yaxis_range + range_pad * np.array([-1, 1]) * np.diff(yaxis_range)
        )
        xaxis_range = tuple(
            xaxis_range + range_pad * np.array([-1, 1]) * np.diff(xaxis_range)
        )

        # transform the p-values so that the important (low) ones are separated out more
        pval_transformed = pval_trans(test_res.pvalues)
        linewidths = [2, 0.5]

        dfs = [
            pd.DataFrame(
                {
                    "pvalue": np.repeat(a=pp, repeats=2),
                    "pvalue_trans": np.repeat(a=ppt, repeats=2),
                    "group": np.array(idxs),
                    "ends": test_res.sample_means[np.array(idxs)],
                    "color": [pal[ii] for ii in idxs],
                    "is_signif": [iss, iss],
                }
            )
            for idxs, pp, ppt, iss in zip(
                self.iterate_pairs(),
                test_res.pvalues,
                pval_transformed,
                test_res.pvalue_is_signif,
            )
        ]
        for ii, df in enumerate(dfs):
            dfs[ii]["midpoint"] = np.repeat(
                np.max(df["ends"]) - 0.5 * np.abs(np.diff(df["ends"])), 2
            )

        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        g = sns.scatterplot(
            data=pd.DataFrame(
                {"mean": test_res.sample_means.mean(), "p-value": [-1, -1]}
            ),
            x="p-value",
            y="mean",
            ax=ax,
        )
        # indicate significance area
        # g.axes.axvspan(axis_range[0], pval_trans(self.alpha), facecolor='lightgray')
        g.axes.axvline(
            x=pval_trans(np.array([self.alpha]))[0],
            ymin=0,
            ymax=1,
            linestyle="dashed",
            color="black",
        )
        # arrow always goes from the first to second
        symbol_dict = {"two-sided": "!=", "less": "<", "greater": ">"}
        title = f"{test_res.alternative} test model A {symbol_dict[test_res.alternative]} model B"
        if test_res.alternative != "two-sided":
            title += "; arrow head goes A -> B"

        for segs in dfs:
            for ii, row in segs.iterrows():
                # g.axes.vlines(x=row['pvalue'], ymin=min(row['ends'], row['midpoint']), ymax=max(row['ends'], row['midpoint']), color=row["color"])
                # if two-sided, each line segment has an arrow starting from the midpoint (xytext) and ending at the endpoint
                # if one-sided, the first row has a segment without an arrow
                arrow_sym = (
                    "->"
                    if ((test_res.alternative == "two-sided") or (ii == 1))
                    else "-"
                )
                plt.annotate(
                    text="",
                    xytext=(row["pvalue"], row["midpoint"]),
                    xy=(row["pvalue"], row["ends"]),
                    arrowprops={
                        "arrowstyle": arrow_sym,
                        "shrinkA": 0,
                        "shrinkB": 0,
                        "color": row["color"],
                        "linewidth": linewidths[0]
                        if row["is_signif"]
                        else linewidths[1],
                    },
                )
        g.set(xlim=xaxis_range, ylim=yaxis_range, title=title)
        g.axes.set_xticks(ticks=pval_trans(tick_vals), labels=tick_vals)
        g.axes.tick_params(right=True, left=True, labelright=True, labelleft=False)

        # label sample means
        for colpal, sm, mn in zip(pal, test_res.sample_means, self.model_names):
            g.text(
                x=xaxis_range[0],
                y=sm,
                s=mn,
                fontdict={"color": colpal, "horizontalalignment": "right"},
            )

        color_legend = [
            Line2D([0], [0], color="red", label=lab, linewidth=lw)
            for lab, lw in zip(["significant", "not significant"], linewidths)
        ]
        color_legend.append(
            Line2D([0], [0], color="black", label="threshold", linestyle="dashed")
        )
        plt.legend(
            handles=color_legend,
            bbox_to_anchor=(1.1, 1),
            loc="upper left",
            fontsize="x-small",
        )
        plt.tight_layout()
        plt.show()

    def _check_valid_signif_report_list(self, test_results_list: list):
        """Test whether a list of results of signif_pair_diff outputs, on different metrics, can be compared.

        Used before running heatmap

        Args:
            test_results_list: List of objects that are results of signif_pair_diff, each on a different metric
        """
        try:
            assert isinstance(test_results_list, list)
            # verify same models are being compared, compare to the base
            for vv in test_results_list:
                self._check_valid_signif_report(vv)

            # all have same correction
            assert (
                len({vv.corrected for vv in test_results_list}) == 1
            ), "must all have same setting of 'corrected'"
            # all have same alternative
            assert (
                len({vv.alternative for vv in test_results_list}) == 1
            ), "must all have same setting of 'alternative"
            # all different metric names
            assert len({vv.metric for vv in test_results_list}) == len(
                test_results_list
            ), "all metric_name must be different"
            assert (
                len({vv.num_obs for vv in test_results_list}) == 1
            ), "must all be performed on samples of the same size"
            assert (
                len({vv.pvalue_alpha for vv in test_results_list}) == 1
            ), "must all have used the same alpha setting"
        except ValueError as e:
            raise ValueError(
                "test_results_list must be a list of signif_pair_diff ouputs on the same models for different metric_names"
            ) from e

    def multiple_metrics_significance_heatmap(
        self,
        test_results_list: list,
        sort_rows=True,
        use_pvalues=True,
        hide_insignificant_rows=False,
        optimize_color=True,
    ):
        """Summarize comparisons of multiple models across at least one metric; all metrics must be done on the same model comparisons.

        Args:
            test_results_list: list where each element corresponds to a different metric result of signif_pair_diff on the same
            set of models compared
            sort_rows: boolean, whether to sort rows so that the most significant comparisons appear at the top
            use_pvalues: boolean, if True use p-values otherwise effect sizes
            hide_insignificant_rows: boolean, if True hide rows (compared pairs) that are not significant for any metric (would be white)
            optimize_color: boolean, if True try to change the order of comparisons (only for two-sided) to maximize the number of
                cells belonging to the majority color
        Returns:

        """
        self._check_valid_signif_report_list(test_results_list)
        alternative = test_results_list[0].alternative
        optimize_color = False if alternative != "two-sided" else optimize_color

        # the heatmap matrices
        hm = self._form_heatmap_matrices(
            test_results_list, sort_rows, use_pvalues, hide_insignificant_rows
        )
        if optimize_color:
            hm = self._optimize_heatmap_color_comparisons(hm, use_pvalues)
        ncomparisons, nmetrics = hm.recoded_values_arr.shape

        # relabel the index by the comparisons
        alt_dict = {
            "two-sided": {"symbol": "vs", "name": "two-sided"},
            "less": {"symbol": "<", "name": "lower-tailed"},
            "greater": {"symbol": ">", "name": "upper-tailed"},
        }
        symb_format = "{} " + alt_dict[alternative]["symbol"] + " {}"

        hm.combined_results.index = [
            symb_format.format(idx[0], idx[1]) for idx in hm.combined_results.index
        ]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        # this coloring uses the recoding, ensure color range is symmetric
        vmx = (
            self.alpha
            if use_pvalues
            else np.nanmax(np.abs(hm.combined_results_arr_with_nan))
        )
        img = ax.matshow(
            hm.recoded_values_arr, vmin=-1 * vmx, vmax=vmx, cmap="bwr_r", aspect="auto"
        )

        # colorbar
        cbar = fig.colorbar(img)

        if use_pvalues:
            # set 5 uniform ticks, which will be from -alpha, with 0 in middle, to alpha
            # do this, rather than setting ticks manually, to ensure that regardless of the choice of alpha, the labeling of ticks is accurate
            cbar.ax.locator_params(nbins=5)
            cbar_ticks = cbar.ax.get_yticklabels()
            # sometimes cbar get_ticks gives ticks outside of the range even though they don't appear on the plot
            cbar_ticks = [
                tt for tt in cbar_ticks if -1 * vmx <= tt.get_position()[1] <= vmx
            ]
            mpt = np.floor(0.5 * len(cbar_ticks)).astype(int)
            # take the upper end of the ticks (including a midpoint right in the middle, since take floor)
            rgt = cbar_ticks[mpt:]
            cbar_tick_pos = np.array([self.alpha - tt.get_position()[1] for tt in rgt])
            cbar_tick_labels = [tt.get_text() for tt in rgt]
            # now reflect
            cbar_tick_labels = cbar_tick_labels + cbar_tick_labels[::-1]
            # positions are the n
            cbar_tick_pos = np.concatenate((-1 * cbar_tick_pos, cbar_tick_pos[::-1]))
            cbar.ax.set_yticks(ticks=cbar_tick_pos, labels=cbar_tick_labels)

        # for effect size, values can be negative or positive, so pad positives with space
        numfmt = "{:.3f}" if use_pvalues else "{: .3f}"

        # label values using the actual p-values not the recoding
        for (j, i), val in np.ndenumerate(hm.combined_results_arr_with_nan):
            if not np.isnan(val):
                # use white if background would be too dark
                ax.text(
                    i,
                    j,
                    numfmt.format(val),
                    ha="center",
                    va="center",
                    fontsize="large",
                    color=("white" if val < 0.5 * vmx else "black")
                    if use_pvalues
                    else ("white" if np.abs(val) > 0.5 * vmx else "black"),
                )

        # label the color bar so it is clear what the blue and red colors meant
        yrg = [-0.5, ncomparisons - 0.5]
        dy = np.diff(yrg)
        cbar_y_pos = [yrg[0] + 0.2 * dy, yrg[1] - 0.2 * dy]
        cbar_x_pos = (nmetrics - 0.5) * 1.02
        cbar_annot = ["1st > 2nd", "1st < 2nd"]
        for yy, txt in zip(cbar_y_pos, cbar_annot):
            ax.annotate(
                text=txt,
                xy=(nmetrics - 1, yy),
                xytext=(cbar_x_pos, yy),
                rotation=90,
                va="center",
            )

        ax.set_xticks(
            ticks=np.arange(hm.combined_results_arr.shape[1]),
            labels=hm.combined_results.columns,
        )
        ax.set_yticks(ticks=np.arange(ncomparisons), labels=hm.combined_results.index)
        ax.set_ylabel("models compared")
        title = "{} model comparison significant {}".format(
            alt_dict[alternative]["name"], "p-values" if use_pvalues else "effect size"
        )
        if hide_insignificant_rows:
            title += "\ncomparisons with no significant differences are omitted"
        plt.title(title)

        plt.table(
            cellText=[[vv] for vv in test_results_list[0].model_names],
            rowLabels=list(range(1, self.nmodels + 1)),
            colLabels=["model name"],
            loc="bottom",
            cellLoc="left",
            colLoc="left",
            edges="open",
        )
        plt.subplots_adjust()  # bottom=0.2)
        plt.show()

    def _form_heatmap_matrices(
        self,
        test_results_list,
        sort_rows=True,
        use_pvalues=True,
        hide_insignificant_rows=False,
    ):
        """Internal function for use in multiple_metrics_significance_heatmap to form the matrices that are displayed."""
        # use 1-indexing for names rather than 0
        combined_results = pd.DataFrame(
            {
                vv.metric: vv.pvalues if use_pvalues else vv.effect_sizes
                for vv in test_results_list
            },
            index=[(a + 1, b + 1) for (a, b) in self.iterate_pairs()],
        )
        # the original p-values only, in array form
        combined_results_arr = combined_results.to_numpy()

        combined_results_are_signif = np.transpose(
            np.vstack(
                [
                    vv.pvalue_is_signif if use_pvalues else vv.effect_size_is_signif
                    for vv in test_results_list
                ]
            )
        )
        combined_results_arr_with_nan = deepcopy(combined_results_arr)
        # set any insignificant results to NaN
        combined_results_arr_with_nan[
            np.logical_not(combined_results_are_signif)
        ] = np.nan

        # color by sign (works for p-values and effect size)
        statistic_sign = pd.DataFrame(
            {
                vv.metric: np.sign(
                    [
                        vv.sample_means[ii] - vv.sample_means[jj]
                        for ii, jj in self.iterate_pairs()
                    ]
                )
                for vv in test_results_list
            }
        )

        if use_pvalues:
            # recode so that high p-values (near alpha) get coded to near 0, and low-pvalues are coded to near alpha, but preserving the sign
            # keep NaNs for coloring
            recoded_values_arr = np.multiply(
                self.alpha - combined_results_arr_with_nan, statistic_sign.to_numpy()
            )
        else:
            # no recoding necessary because value of effect size already shows directly
            recoded_values_arr = combined_results_arr_with_nan

        if sort_rows:
            if use_pvalues:
                # geometric mean p-value regardless of significance, lower values are more significant
                score_ord = np.argsort(
                    np.apply_along_axis(arr=combined_results_arr, axis=1, func1d=gmean)
                )
            else:
                # take absolute value (to ignore sign) first then reverse since higher values are more significant
                score_ord = np.argsort(np.mean(np.abs(combined_results_arr), axis=1))[
                    ::-1
                ]

            recoded_values_arr = recoded_values_arr[score_ord, :]
            combined_results_arr_with_nan = combined_results_arr_with_nan[score_ord, :]
            combined_results = combined_results.iloc[score_ord]

        if hide_insignificant_rows:
            # if any of results are not NaN (significant), then include
            row_not_all_nan = np.any(
                a=np.logical_not(np.isnan(combined_results_arr_with_nan)), axis=1
            )
            recoded_values_arr = recoded_values_arr[row_not_all_nan, :]
            combined_results_arr_with_nan = combined_results_arr_with_nan[
                row_not_all_nan, :
            ]
            combined_results = combined_results.loc[row_not_all_nan]

        return self.HEATMAP_MATRICES(
            recoded_values_arr,
            combined_results_arr_with_nan,
            combined_results,
            combined_results_arr,
        )

    def _optimize_heatmap_color_comparisons(self, hm, use_pvalues):
        """Internal function for use in multiple_metrics_significance_heatmap to optimize the color choice to so most cells have the same color (red/blue)."""
        assert isinstance(hm, self.HEATMAP_MATRICES)

        from scipy.optimize import minimize

        (
            recoded_values_arr,
            combined_results_arr_with_nan,
            combined_results,
            combined_results_arr,
        ) = deepcopy(hm)

        row_not_all_nan = np.any(
            a=np.logical_not(np.isnan(combined_results_arr_with_nan)), axis=1
        )
        recoded_sign = np.sign(recoded_values_arr[row_not_all_nan, :])
        ncomparisons, nmetrics = combined_results_arr_with_nan.shape
        ncomparisons_not_nan = row_not_all_nan.sum()
        # default is not to change
        change_comparison_order = np.zeros(ncomparisons).astype(bool)

        if ncomparisons_not_nan > 0:
            recoded_sign = recoded_values_arr[row_not_all_nan, :]

            # sign corresponds to colors; sum of signs is the difference between the number of blue and red cells in the row.
            # a higher gap means there is more potentially to exploit if we change the order, which will flip the colors
            # take sum of change indicator (0 -> -0.5, 1 -> 0.5) to make opposite sign
            # sum across rows, then take absolute value (so red/blue are treated the same), multiply by -1 then minimize to maximize the gap
            def diff_each_colors(x):
                return -1 * np.abs(
                    sum(
                        [
                            (ind - 0.5) * np.nansum(rvals)
                            for ind, rvals in zip(x, recoded_sign)
                        ]
                    )
                )

            ores = minimize(
                fun=diff_each_colors,
                x0=np.zeros(ncomparisons_not_nan),
                bounds=tuple([(0, 1) for rr in range(ncomparisons_not_nan)]),
            )
            change_comparison_order[row_not_all_nan] = np.round(ores.x).astype(bool)

            # reorder index if changed
            combined_results.index = [
                (idx[1], idx[0]) if chg else idx
                for idx, chg in zip(combined_results.index, change_comparison_order)
            ]
            # change sign of recoded values
            recoded_values_arr = np.vstack(
                [
                    -1 * row if chg else row
                    for row, chg in zip(recoded_values_arr, change_comparison_order)
                ]
            )

            # if use effect size, need to change the sign of the printed values as well
            if not use_pvalues:
                combined_results_arr_with_nan = np.vstack(
                    [
                        -1 * row if chg else row
                        for row, chg in zip(
                            combined_results_arr_with_nan, change_comparison_order
                        )
                    ]
                )
                combined_results = combined_results.multiply(
                    other=[-1 if chg else 1 for chg in change_comparison_order], axis=0
                )

        # only combined_results (the dataframe, with its index), combined_results_arr_with_nan
        return self.HEATMAP_MATRICES(
            recoded_values_arr,
            combined_results_arr_with_nan,
            combined_results,
            combined_results_arr,
        )

    def metric_significant_pairs_graph(
        self,
        test_res,
        node_color_levels=None,
        use_pvalues=True,
        model_name_split_char=None,
        weight_edges=False,
    ):
        """Compare models across a given metric; show a graph where nodes correspond to models, and edges are drawn between pairs that have a significant difference result.

        Args:
            test_res: object outputted from signif_pair_diff
            node_color_levels: if not None, list of levels corresponding to model names to use for coloring nodes;
                e.g., node_color_levels=['A', 'B', 'A']} means the 1st and 3rd nodes will have one color and the 2nd a different one
            use_pvalues: boolean, if True use p-values, otherwise effect sizes, to decide if significant
            model_name_split_char: string used to split node names on when labeling
            weight_edges: boolean, if True make less significant differences be shown by thicker edge lines
        """
        self._check_valid_signif_report(test_res)
        if node_color_levels is not None:
            assert len(node_color_levels) == self.nmodels

        criterion = "pvalue" if use_pvalues else "effect_size"

        # form the graph and set the edges and node positions
        dg = self._form_connected_graph(
            test_res, criterion, model_name_split_char, weight_edges
        )
        cleg = self._graph_color_legend(dg, node_color_levels)

        # plot the figure
        import matplotlib.pyplot as plt
        import networkx as nx

        fig, ax = plt.subplots(1, 1)
        nx.draw_networkx(
            dg.graph,
            with_labels=True,
            pos=dg.positions,
            ax=ax,
            node_color=cleg.node_color_vec,
            arrows=test_res.alternative != "two-sided",
            edge_color="gray",
            width=dg.edge_widths,
        )
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
        ax.set_ylabel(f"mean model {test_res.metric}")
        # add some space to x limits so label names don't hit the plotting panel edges
        xlims = ax.get_xlim()
        # retrieve plotted positions, one for each node (model)
        x_coords = [cc[0] for cc in dg.positions.values()]
        x_margin = (max(x_coords) - min(x_coords)) * 0.25
        ax.set_xlim(xlims[0] - x_margin, xlims[1] + x_margin)
        direction = {"two-sided": "A vs B", "less": "A < B", "greater": "A > B"}
        title = "{}: edges connect models with insignificant {} {}".format(
            test_res.metric,
            direction[test_res.alternative],
            criterion.replace("_", " "),
        )
        ax.set_title(
            title + "\nthicker lines mean a less significant difference"
            if weight_edges
            else title
        )

        if cleg.color_legend is not None:
            plt.legend(
                handles=cleg.color_legend, bbox_to_anchor=(1.01, 1), loc="upper left"
            )
            plt.tight_layout()
        plt.show()

    def _make_graph(self, test_res, criterion):
        """Internal function to metric_significant_pairs_graph to form the connected graph nodes and edges."""
        # add nodes and edges
        edges = {}
        for pair, is_signif, signif in zip(
            self.iterate_pairs(),
            getattr(test_res, f"{criterion}_is_signif"),
            getattr(test_res, f"{criterion}s"),
        ):
            if not is_signif:
                if pair[0] not in edges:
                    edges[pair[0]] = {"to": [], "weight": []}
                # only plot not significant connections
                edges[pair[0]]["to"].append(pair[1])
                edges[pair[0]]["weight"].append(np.abs(signif))

        # random seed so positions results are the same across runs
        import networkx as nx

        sd = 5
        np.random.seed(sd)
        g = nx.DiGraph()
        for node, others in edges.items():
            if len(others["to"]):
                for ii, jj in enumerate(others["to"]):
                    g.add_edge(node, jj, weight=others["weight"][ii])
            else:
                g.add_node(node)

        for node in range(self.nmodels):
            # add lone nodes that aren't connected to any others
            if node not in g.nodes:
                g.add_node(node)

        # change vertical position only so aligns with the height of the sample mean (the average metric value)
        pos = nx.spring_layout(g, seed=sd, k=10 / np.sqrt(g.order()))
        pos = {
            ii: np.array([pos[ii][0], sm])
            for ii, sm in enumerate(test_res.sample_means)
        }
        return g, pos

    def _form_connected_graph(
        self, test_res, criterion, model_name_split_char=None, weight_edges=False
    ):
        """Internal function to metric_significant_pairs_graph to form the connected graph displayed."""
        import networkx as nx

        g, pos = self._make_graph(test_res, criterion)

        # and relabel the nodes with the model names
        model_names = deepcopy(self.model_names)
        if model_name_split_char is not None:
            model_names = [
                mn.replace(model_name_split_char, "\n") for mn in model_names
            ]

        # nodes correspond to models
        nodeidx2name = dict(zip(range(self.nmodels), model_names))
        # rename the keys
        pos = {mn: pos[ii] for ii, mn in nodeidx2name.items()}
        g = nx.relabel_nodes(g, mapping=nodeidx2name)
        # widths
        if weight_edges is False:
            # the default width
            edge_widths = [1.0] * self.nmodels
        else:
            # range of edge widths to draw
            width_range = [0.5, 4]
            signif_ranges = {"pvalue": [self.alpha, 1], "effect_size": [0.8, 5]}

            # translate the significance values into weights
            weights = np.clip(
                a=[g[i][j]["weight"] for i, j in g.edges()],
                a_min=signif_ranges[criterion][0],
                a_max=signif_ranges[criterion][1],
            )
            edge_widths = width_range[0] + (
                weights - signif_ranges[criterion][0]
            ) * np.diff(width_range) / np.diff(signif_ranges[criterion])

        DIRECTED_GRAPH = namedtuple(
            "DirectedGraph", "graph positions edge_widths node_mapping"
        )
        return DIRECTED_GRAPH(g, pos, edge_widths, nodeidx2name)

    def _graph_color_legend(self, dg, node_color_levels=None):
        """Internal function to create a color legend and determine the node colors."""
        # reverse the key-value mapping
        node2index = {node: ii for ii, node in dg.node_mapping.items()}

        if node_color_levels is not None:
            # unique number of categories
            nlevels = len(set(node_color_levels))
            if nlevels > 1:
                pal = spectral_palette(n=nlevels)
                # reorder levels by node order
                level2node = defaultdict(list)
                # means value val in color_node_levels is associated with the ith models, but first sort in alphabetical order
                for ii, val in enumerate(node_color_levels):
                    level2node[val].append(ii)
                colors = [[] for _ in range(self.nmodels)]
                for ii, val in enumerate(sorted(level2node)):
                    for node in level2node[val]:
                        # take the ith color
                        colors[node] = pal[ii]
                # now reorder color vec by the order in which the nodes are stored internally in the graph (not necessarily in numberic order)
                node_color_levels_vec = [
                    colors[node2index[node]] for node in dg.graph.nodes
                ]

                from matplotlib.lines import Line2D

                color_legend = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=lab,
                        markerfacecolor=pal[ii],
                        markersize=12,
                    )
                    for ii, lab in enumerate(sorted(level2node))
                ]
            else:
                # default same color for all
                node_color_levels_vec = ["lightblue"] * self.nmodels
                color_legend = None
        else:
            node_color_levels_vec = ["lightblue"] * self.nmodels
            color_legend = None

        GRAPH_COLOR_LEGEND = namedtuple("GraphLegend", "color_legend node_color_vec")
        return GRAPH_COLOR_LEGEND(color_legend, node_color_levels_vec)
