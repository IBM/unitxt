from unitxt.metric_utils import SubsetsScores

from tests.utils import UnitxtTestCase


class TestSerializers(UnitxtTestCase):
    def test_subset_scores_summary(self):
        subset_scores = SubsetsScores(
            {
                "stsb": {
                    "classification": {
                        "num_of_instances": 5,
                        "spearmanr": -0.968,
                        "score": -0.968,
                        "score_name": "spearmanr",
                        "score_ci_low": -1.0,
                        "score_ci_high": -0.745,
                        "spearmanr_ci_low": -1.0,
                        "spearmanr_ci_high": -0.745,
                        "groups": {
                            "template": {
                                "templates.regression.two_texts.title": {
                                    "num_of_instances": 4,
                                    "spearmanr": -0.943,
                                    "score": -0.943,
                                    "score_name": "spearmanr",
                                    "score_ci_low": -1.0,
                                    "score_ci_high": -0.816,
                                    "spearmanr_ci_low": -1.0,
                                    "spearmanr_ci_high": -0.816,
                                },
                                "templates.regression.two_texts.similarity.flan": {
                                    "num_of_instances": 1,
                                    "spearmanr": None,
                                    "score": None,
                                    "score_name": "spearmanr",
                                },
                            }
                        },
                    },
                    "regression": {
                        "num_of_instances": 8,
                        "spearmanr": -0.065,
                        "score": -0.065,
                        "score_name": "spearmanr",
                        "score_ci_low": -0.784,
                        "score_ci_high": 0.711,
                        "spearmanr_ci_low": -0.784,
                        "spearmanr_ci_high": 0.711,
                    },
                    "score": -0.517,
                    "score_name": "subsets_mean",
                    "num_of_instances": 13,
                },
                "wnli": {
                    "num_of_instances": 12,
                    "f1_macro": 0.357,
                    "f1_entailment": 0.0,
                    "f1_not entailment": 0.714,
                    "score": 0.5,
                    "score_name": "f1_micro",
                    "score_ci_low": 0.235,
                    "score_ci_high": 0.736,
                    "f1_macro_ci_low": 0.205,
                    "f1_macro_ci_high": 0.429,
                    "accuracy": 0.417,
                    "accuracy_ci_low": 0.167,
                    "accuracy_ci_high": 0.667,
                    "f1_micro": 0.5,
                    "f1_micro_ci_low": 0.235,
                    "f1_micro_ci_high": 0.736,
                    "groups": {
                        "template": {
                            "templates.classification.multi_class.relation.default": {
                                "num_of_instances": 12,
                                "f1_macro": 0.357,
                                "f1_entailment": 0.0,
                                "f1_not entailment": 0.714,
                                "score": 0.5,
                                "score_name": "f1_micro",
                                "score_ci_low": 0.235,
                                "score_ci_high": 0.736,
                                "f1_macro_ci_low": 0.205,
                                "f1_macro_ci_high": 0.429,
                                "accuracy": 0.417,
                                "accuracy_ci_low": 0.167,
                                "accuracy_ci_high": 0.667,
                                "f1_micro": 0.5,
                                "f1_micro_ci_low": 0.235,
                                "f1_micro_ci_high": 0.736,
                            }
                        }
                    },
                },
                "rte": {
                    "num_of_instances": 5,
                    "f1_macro": 0.333,
                    "f1_not entailment": 0.667,
                    "f1_entailment": 0.0,
                    "score": 0.5,
                    "score_name": "f1_micro",
                    "score_ci_low": 0.0,
                    "score_ci_high": 0.795,
                    "f1_macro_ci_low": 0.0,
                    "f1_macro_ci_high": 0.75,
                    "accuracy": 0.4,
                    "accuracy_ci_low": 0.0,
                    "accuracy_ci_high": 0.8,
                    "f1_micro": 0.5,
                    "f1_micro_ci_low": 0.0,
                    "f1_micro_ci_high": 0.795,
                },
                "score": 0.161,
                "score_name": "subsets_mean",
                "num_of_instances": 30,
            }
        )

        self.assertTrue(subset_scores.summary)
