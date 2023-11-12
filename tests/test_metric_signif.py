import numpy as np

import unittest
from src.unitxt.metric_paired_significance import PairedDifferenceTest


class TestMetricSignifDifference(unittest.TestCase):
    @classmethod
    def setUpClass(cls, nmodels=4, nobs=50):
        cls.nmodels = max(2, int(nmodels))
        cls.nobs = max(3, int(nobs))

    def gen_continuous_data(self, same_distr=True):
        # assume we have a dataset with nobs observations
        # generate a matrix of size (nmodels, nobs) representing the results of nmodels results on the same nobs observations
        # same_distr means they follow the same distribution
        # covariance matrix to generate observations that are paired.  Each have correlation 0.7 and variance 1
        cmat = np.empty((self.nmodels, self.nmodels))
        cmat.fill(0.7)
        np.fill_diagonal(cmat, 1)

        mu = np.random.normal(size=self.nobs, loc=5)
        model_measurement = np.transpose(np.vstack([np.random.multivariate_normal(mean=[mm]*self.nmodels, cov=cmat) for mm in mu]))
        if not same_distr:
            # make the last two sample have a higher mean
            model_measurement[-1, :] = model_measurement[-1, :] + 2
            if self.nmodels > 2:
                model_measurement[-2, :] = model_measurement[-2, :] + 1

        # add some skew
        model_measurement = np.square(model_measurement)
        return tuple([xx for xx in model_measurement])

    
    def gen_binary_data(self, same_distr=True, nmodels=None):
        # generate only binary data
        nmodels = self.nmodels if nmodels is None else max(2, int(nmodels))
        if same_distr:
            # generate random probabilities for each observation and then binary
            # do this so observation pairs are more correlated than otherwise if used the same p for all
            p = np.random.beta(a=2, b=5, size=self.nobs)
            return [np.random.binomial(n=1, p=p) for rr in range(nmodels)]
        else:
            p = np.vstack([np.random.beta(a=2, b=5, size=(nmodels-1, self.nobs)), np.random.beta(a=5, b=2, size=(1, self.nobs))])
            return [np.random.binomial(n=1, p=pp) for pp in p]

    
    def _test_signif(self, expected_pvalues_list: list, expected_effect_sizes, same_distr=True, continuous=True):
        np.random.seed(4)

        model_res = self.gen_continuous_data(same_distr) if continuous else self.gen_binary_data(same_distr=same_distr)
        tester = PairedDifferenceTest(nmodels=self.nmodels)

        # use default paired t-test
        res_twosided = tester.signif_pair_diff(samples_list=model_res, alternative='two-sided')
        for observed, expected in zip(res_twosided.pvalues, expected_pvalues_list[0]):
            self.assertAlmostEqual(first=observed, second=expected)
        # the effect sizes are the same in the one and two-sided case, and only the non-permutation case
        for observed, expected in zip(res_twosided.effect_sizes, expected_effect_sizes):
            self.assertAlmostEqual(first=observed, second=expected)

        res_onesided = tester.signif_pair_diff(samples_list=model_res, alternative='less')
        for observed, expected in zip(res_onesided.pvalues, expected_pvalues_list[1]):
            self.assertAlmostEqual(first=observed, second=expected)

        # permutation results should be very similar to t-test but not identical, and should vary a bit each run due to permutation randomness
        res_twosided = tester.signif_pair_diff(samples_list=model_res, alternative='two-sided', permute=True)
        for observed, expected in zip(res_twosided.pvalues, expected_pvalues_list[2]):
            self.assertAlmostEqual(first=observed, second=expected)
        res_onesided = tester.signif_pair_diff(samples_list=model_res, alternative='less', permute=True)
        for observed, expected in zip(res_onesided.pvalues, expected_pvalues_list[3]):
            self.assertAlmostEqual(first=observed, second=expected)

    def test_signif_same_distr_continuous(self):
        self._test_signif(expected_pvalues_list=[np.array([0.85442384, 0.90404157, 0.826577, 0.826577, 0.90404157, 0.72593365]),
                                                 np.array([0.66103154, 0.88089618, 0.59450227, 0.88089618, 0.82079927, 0.4579273]),
                                                 np.array([0.86229215, 0.90611904, 0.82509877, 0.82509877, 0.90611904, 0.6937621]),
                                                 np.array([0.66001193, 0.88083696, 0.59651234, 0.88083696, 0.8209753, 0.44840733])],
                          expected_effect_sizes=np.array([-0.27137964,  0.15076678, -0.36974064,  0.39757755, -0.06070158, -0.49517756]),
                          same_distr=True)

    def test_signif_diff_distr_continuous(self):
        # here the last one or two samples (models) have higher mean, so the 'less' alternative should be appropriate
        self._test_signif(expected_pvalues_list=[np.array([4.73946291e-01, 2.13890672e-10, 2.46099435e-23, 1.06904139e-09, 7.72220177e-20, 9.73759291e-13]),
                                                 np.array([2.36973146e-01, 1.06945336e-10, 1.23049717e-23, 5.34520693e-10, 3.86110089e-20, 4.86879645e-13]),
                                                 np.array([0.4836, 0.0011994, 0.0011994, 0.0011994, 0.0011994, 0.0011994]),
                                                 np.array([0.2364, 0.00059985, 0.00059985, 0.00059985, 0.00059985, 0.00059985])],
                          expected_effect_sizes=np.array([-0.27137964, -3.11411273, -7.09385033, -2.89886275, -5.83322115, -3.73910222]),
                          same_distr=False)

    def test_signif_same_distr_binary(self):
        # use ordinary t-test or permutation on binary values; have more than 2 samples so don't use McNemar
        self._test_signif(expected_pvalues_list=[np.array([0.43845116, 0.72657074, 0.72657074, 0.17494627, 0.72657074, 0.50067149]),
                                                 np.array([0.28555526, 0.98904844, 0.68091409, 0.98904844, 0.98904844, 0.33981817]),
                                                 np.array([0.60324019, 0.86656717, 0.86656717, 0.28055268, 0.86656717, 0.65894208]),
                                                 np.array([0.42384358, 0.99636195, 0.79740386, 0.99636195, 0.99636195, 0.4785349])],
                          expected_effect_sizes=np.array([-0.61389088,  0.29011333, -0.25744353,  0.83244498,  0.35415175, -0.53734092]),
                          same_distr=True, continuous=False)

    def test_signif_diff_distr_binary(self):
        # use ordinary t-test or permutation on binary values; have more than 2 samples so don't use McNemar
        # here the last one or two samples (models) have higher mean, so the 'less' alternative should be appropriate
        self._test_signif(expected_pvalues_list=[np.array([0.87261054, 0.87261054, 0.01301297, 0.87261054, 0.0031221, 0.00107259]),
                                                 np.array([9.30778351e-01, 9.30778351e-01, 6.52246273e-03, 9.30778351e-01, 1.56202768e-03, 5.36416862e-04]),
                                                 np.array([1.        , 0.95056914, 0.02299894, 0.96095424, 0.0019984, 0.0011994 ]),
                                                 np.array([0.96683102, 0.96683102, 0.01471806, 0.96683102, 0.00499001, 0.00299625])],
                          expected_effect_sizes=np.array([0.08545204,  0.25744353, -1.1630991 ,  0.18660746, -1.37473764, -1.52511603]),
                          same_distr=False, continuous=False)

    def test_signif_mcnemar_binary(self):
        np.random.seed(4)

        # use Mcnemar's test, not t-test, only on two model samples
        tester = PairedDifferenceTest(nmodels=2)

        # random generation of paired binary data
        binary_same = self.gen_binary_data(same_distr=True, nmodels=tester.nmodels)
        res = tester.signif_pair_diff(samples_list=binary_same)
        self.assertAlmostEqual(first=res.pvalues[0], second=0.1670684814453125)
        self.assertAlmostEqual(first=res.effect_sizes[0], second=0.1842105263157895)

        binary_diff = self.gen_binary_data(same_distr=False, nmodels=tester.nmodels)
        res = tester.signif_pair_diff(samples_list=binary_diff)
        self.assertAlmostEqual(first=res.pvalues[0], second=0.028959274291992188)
        self.assertAlmostEqual(first=res.effect_sizes[0], second=0.23076923076923073)

        # handle some corner cases where the samples do not result in a 2x2 contingency table automatically
        # contingency table is 1x1
        samples_list = [np.ones(shape=100), np.ones(shape=100)]
        res_1x1 = tester.signif_pair_diff(samples_list=samples_list)
        # p-value is 1 meaning there is no difference since both have same values exactly, with no variability
        self.assertAlmostEqual(first=res_1x1.pvalues[0], second=1.0)
        self.assertAlmostEqual(first=res_1x1.effect_sizes[0], second=0.0)

        # also 1x1 but different values 0 and 1
        samples_list = [np.ones(shape=100), np.zeros(shape=100)]
        res_1x1 = tester.signif_pair_diff(samples_list=samples_list)
        # p-value is essentially 0 because there is a complete difference and no variability
        self.assertAlmostEqual(first=res_1x1.pvalues[0], second=1.5777218104420236e-30)
        self.assertAlmostEqual(first=res_1x1.effect_sizes[0], second=0.49502487562189057)

        # contingency table is 1x2
        samples_list = [np.ones(shape=100), np.repeat(a=[0, 1], repeats=[47, 53])]
        res_1x2 = tester.signif_pair_diff(samples_list=samples_list)
        self.assertAlmostEqual(first=res_1x2.pvalues[0], second=1.4210854715202004e-14)
        self.assertAlmostEqual(first=res_1x2.effect_sizes[0], second=0.4894736842105263)

        # contingency table is 2x1
        samples_list = [np.repeat(a=[0, 1], repeats=[49, 51]), np.zeros(shape=100)]
        res_2x1 = tester.signif_pair_diff(samples_list=samples_list)
        self.assertAlmostEqual(first=res_2x1.pvalues[0], second=8.881784197001252e-16)
        self.assertAlmostEqual(first=res_2x1.effect_sizes[0], second= 0.49029126213592233)

        # contingency table is 2x2 but with one combination missing, which needs a continuity correction
        samples_list = [np.repeat(a=[0,1], repeats=[50,50]), np.repeat(a=[0,1], repeats=[40,60])]
        res_2x2 = tester.signif_pair_diff(samples_list=samples_list)
        self.assertAlmostEqual(first=res_2x2.pvalues[0], second=0.001953125)
        self.assertAlmostEqual(first=res_2x2.effect_sizes[0], second=0.45238095238095233)


    # def test_lineplot(self, alternative='two-sided'):
    #     np.random.seed(6)
    #     model_res = self.gen_continuous_data(same_distr=False)
    #     tester = PairedDifferenceTest(nmodels=self.nmodels)
    #     test_res = tester.signif_pair_diff(samples_list=model_res, alternative=alternative)
    #     tester.lineplot(test_res=test_res)

