import numpy as np
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from scipy.stats import permutation_test, ttest_rel, norm
# import seaborn as sns
import pandas as pd
# import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

predictions = ["A", "B", "C"]
references = [["B"], ["A"], ["C"]]

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


class PairedDifferenceTest:
    """
    Class to conduct test of statistical significance (from 0) in average differences between
    sample values for the same observation index.
    Can be used to measure if a pair of prediction models differ substantially in the metric scores they
    receive across observations.
    """
    def __init__(self, nmodels=2, alpha=0.05):
        """
        Args:
            nmodels: the number of models (samples) to be compared pairwise
            alpha: statistical false positive rate confidence to be used in evaluating significance
        """
        self.nmodels = max(2, int(nmodels))
        # desired false positive rate
        self.alpha = float(np.clip(alpha, a_min=1e-10, a_max=0.5))
        self.binary_values = np.array([0,1])

    @staticmethod
    def _permute_diff_statistic(lx, ly):
        """
        Statistic to be used in permutation test.  Calculates expected difference between pairs (= difference
        in expected values of the samples themselves)
        Args:
            lx: first sample
            ly: second sample
        """
        return np.mean(lx) - np.mean(ly)

    def _check_valid_samples(self, samples_list):
        '''
        Check if samples_list is valid (is of length nmodels), all have same length, and at least 2
        Args:
            samples_list: a list of 1-D numpy arrays
        '''
        # list of equal-length samples greater than size 2
        len_elements = np.unique([len(vec) for vec in samples_list])
        assert (len(samples_list) == self.nmodels) and (len(np.unique(len_elements)) == 1) and (len_elements[0] >= 2)


    def _check_binary(self, samples_list):
        """
        Check if samples are binary-valued
        Args:
            samples_list: a list of 1-D numpy arrays

        Returns:
            boolean
        """
        # return True/False if binary, and returns the binary values
        self._check_valid_samples(samples_list)
        uv = np.unique(np.vstack(samples_list))
        return np.all(np.isin(uv, self.binary_values))

    def _can_use_mcnemar(self, samples_list, alternative='two-sided'):
        """
        Check if can use the McNemar test (accepts only 2 binary equal-length samples) which simplifies computation
        Args:
            samples_list: a list of 1-D numpy arrays
            alternative: alternative hypothesis to be used (only two-sided is valid)

        Returns:
            boolean
        """
        is_binary = self._check_binary(samples_list)
        return is_binary and (len(samples_list) == 2) and (alternative == 'two-sided')

    def _handle_binary_data_contingency(self, samples_list):
        """
        Convert samples into 2x2 contingency table form, even if some value combinations are not observed
        Args:
            samples_list: a list of 1-D numpy arrays

        Returns:
            2x2 numpy cross-tabulation array
        """
        # make sure that a cross tabulation is done that is 2x2 in case some value combinations are missing
        cat_binary = pd.CategoricalDtype(categories=self.binary_values, ordered=False)
        # encode as categorical binary with all values, use dropna=False to avoid dropping missing combinations
        df = pd.DataFrame(np.transpose(np.vstack(samples_list))).astype(cat_binary)
        return pd.crosstab(df[0], df[1], dropna=False).to_numpy()



    def mcnemar_test(self, samples_list):
        """
        Perform McNemar test and return a p-value
        Args:
            samples_list: a list of 1-D numpy arrays (must be binary)

        Returns:
            float
        """
        # assertion in case is used outside of signif_pair_diff
        assert(len(samples_list) == 2, "McNemar's test can use only two samples")
        return mcnemar(table=self._handle_binary_data_contingency(samples_list), exact=True).pvalue


    def iterate_pairs(self, samples_list=None):
        """
        Iterate over ordered combinations of samples or their indices (if samples_list is None)
        Args:
            samples_list: a list of 1-D numpy arrays (must be binary)

        Returns:
            iterator
        """
        # return pairs of values (samples) that are compared, or the indices if samples_list=None
        # notice, the pair order is important if one-sided tests are used because it affects the hypothesis order
        return combinations(range(self.nmodels) if samples_list is None else samples_list, r=2)

    def signif_pair_diff(self, samples_list, alternative='two-sided', permute=False, corrected=True):
        """
        Conduct pairedd-observation est of difference in means between samples
        Args:
            samples_list: a list of 1-D numpy arrays
            alternative: alternative hypothesis to be used; one-sided (greater/less) should only be used if the arrays have been ordered
            permute: whether or not to use permutation test (requires some simulation)
            corrected: whether or not to apply a correction for control of false positive rate given the number of hypotheses (used when nmodels > 2)

        Returns:
            dict
        """
        # because sample ordering is not fixed, do either greater (one-sided) or two-sided
        assert alternative in ('greater', 'less', 'two-sided')
        if alternative != 'two-sided':
            flds = ('greater', 'descending') if alternative == 'greater' else ('less', 'ascending')
            print("If 'alternative' is '{}, the samples are expected to be sorted in {} order of ANTICIPATED (not OBSERVED) mean value".format(flds[0], flds[1]))
        if self.nmodels == 2:
            # require more than 2 hypotheses (only if nmodels > 2) to be able to do a correction
            corrected = False

        # first see if can use McNemar's test, better for binary data; only works for two-sided hypotheses with two samples
        # validity check
        if self._can_use_mcnemar(samples_list, alternative):
            pvalue = self.mcnemar_test(samples_list=samples_list)
            permute = False
            res = {"pvalues": np.array([pvalue]), "is_signif": np.array([pvalue <= self.alpha])}

        else:
            # most other cases if have continuous variables or more than 2 models to compare
            # greater when ttest_rel(a,b) checks if a_i - b_i > 0
            # note, regardless of the significance, the value of the paired expectation E(a_i - b_i) will be the same as the
            # difference in means, i.e. E(a) - E(b).

            # pvalue is low (close to 0) if E(a_i - b_i) is significant (according to altnernative)
            if permute:
                pvalues = np.array([permutation_test(data=vec, statistic=self._permute_diff_statistic, alternative=alternative, permutation_type='samples', vectorized=False).pvalue
                                    for vec in self.iterate_pairs(samples_list=samples_list)])
            else:
                pvalues = np.array([ttest_rel(a=vec[0], b=vec[1], alternative=alternative).pvalue for vec in self.iterate_pairs(samples_list=samples_list)])

            if corrected:
                mult_corr = multipletests(pvals=pvalues, alpha=self.alpha)
                res = {'pvalues': mult_corr[1], "is_signif": mult_corr[0]}
            else:
                res = {"pvalues": pvalues, "is_signif": pvalues <= self.alpha}

        res.update({"corrected": corrected, "alternative": alternative, "permute": permute, 'sample_means': np.array([np.mean(vec) for vec in samples_list])})

        return res



    # def lineplot(self, test_res: dict):
    #     assert all([kk in test_res for kk in ('pvalues', 'alternative', 'permute', 'corrected', 'is_signif', 'sample_means')])
    #     pal = sns.color_palette(n_colors=self.nmodels, as_cmap=True)
    #     level_order = np.argsort(test_res['sample_means'])[::-1]
    #     # transform the p-values
    #     pval_transformed = pval_trans(test_res['pvalues'])
    #     tick_vals = np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.5, 0.1, 0.2, 0.5, 1.0])
    #     pval_range = np.array([test_res['pvalues'].min(), test_res['pvalues'].max()])
    #     tick_vals = np.array([vv for vv in tick_vals if vv >= pval_range[0] and vv <= pval_range[1]])
    #     axis_range = tuple(pval_trans(pval_range) + 0.03 * np.array([-1,1]))
    #
    #     pal = np.array([pal[ii] for ii in level_order])
    #     dfs = [pd.DataFrame({'pvalue': np.repeat(a=pp, repeats=2),
    #                          'group': np.array(idxs),
    #                          'ends': test_res['sample_means'][np.array(idxs)], 'color': pal[np.array(idxs)]}).sort_values(by=['ends'])
    #                         for idxs, pp in zip(self.iterate_pairs(), pval_transformed)]
    #     for ii, df in enumerate(dfs):
    #         dfs[ii]["midpoint"] = np.repeat(np.max(df['ends']) - 0.5 * np.abs(np.diff(df['ends'])), 2)
    #
    #     g = sns.scatterplot(data=pd.DataFrame({"mean": test_res['sample_means'].mean(), "p-value": [-1, -1]}),
    #                         x="p-value", y="mean")
    #     for segs in dfs:
    #         for _, row in segs.iterrows():
    #             # apply visualization transformation
    #             g.axes.vlines(x=row['pvalue'], ymin=min(row['ends'], row['midpoint']), ymax=max(row['ends'], row['midpoint']), color=row["color"])
    #     g.set(xlim=axis_range, ylim=(test_res['sample_means'].min(), test_res['sample_means'].max()))
    #     g.axes.set_xticks(ticks=pval_trans(tick_vals), labels=tick_vals)




    # def signif_report(self, samples_list, alternative="two-sided", permute=False, corrected=True):
    #     # return only the pairs that are significant, and their pair indices
    #     test_res = self.signif_pair_diff(samples_list, alternative, permute, corrected)
    #     res = [(pairs, p) for pairs, p, is_signif in zip(self.iterate_pairs(), test_res["pvalues"], test_res["is_signif"]) if is_signif]
    #     res = {"pairs": [val[0] for val in res], "pvalue": [val[1] for val in res]}
    #     res["alternative"] = alternative
    #     return res

