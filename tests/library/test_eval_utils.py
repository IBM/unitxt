import pandas as pd
from unitxt.eval_utils import evaluate

from tests.utils import UnitxtTestCase


class TestEvalUtils(UnitxtTestCase):
    df = pd.DataFrame(
        [[1.0, [0.0]], [0.0, [1.0]], [0.0, [0.0]]], columns=["prediction", "references"]
    )
    metrics = ["metrics.accuracy", "metrics.spearman"]

    def test_evaluate(self):
        results_df, global_scores = evaluate(self.df, self.metrics)
        results_df = results_df.round(2)

        results2, global_scores2 = evaluate(self.df.to_dict("records"), self.metrics)
        results2_df = pd.DataFrame(results2).round(2)

        for metric in self.metrics:
            self.assertSequenceEqual(
                list(results_df[metric].astype(str)),
                list(results2_df[metric].astype(str)),
                3,
            )
            self.assertAlmostEqual(
                global_scores[metric]["score"], global_scores2[metric]["score"], 3
            )

        self.assertSequenceEqual(list(results_df["metrics.accuracy"]), [0.0, 0.0, 1.0])
        self.assertAlmostEqual(global_scores["metrics.accuracy"]["score"], 0.3333, 3)
        self.assertAlmostEqual(global_scores["metrics.spearman"]["score"], -0.5, 3)

        results_df, global_scores_df = evaluate(
            self.df, ["metrics.accuracy"], compute_conf_intervals=True
        )
        global_scores = global_scores_df["metrics.accuracy"].to_dict()
        self.assertDictEqual(
            global_scores,
            {
                "accuracy": 0.3333333333333333,
                "score": 0.3333333333333333,
                "score_name": "accuracy",
                "accuracy_ci_low": 0.0,
                "accuracy_ci_high": 1.0,
                "score_ci_low": 0.0,
                "score_ci_high": 1.0,
            },
        )
