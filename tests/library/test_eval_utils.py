import os

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
                "num_of_instances": 3,
            },
        )

    def test_evaluate_with_llmaaj_and_external_client(self):
        from ibm_watsonx_ai.client import APIClient, Credentials
        from unitxt.inference import WMLInferenceEngineChat
        from unitxt.llm_as_judge_from_template import TaskBasedLLMasJudge

        external_client=APIClient(
            credentials=Credentials(
                api_key=os.environ.get("WML_APIKEY"), url=os.environ.get("WML_URL")
            ),
            project_id=os.environ.get("WML_PROJECT_ID")
        )

        metric = TaskBasedLLMasJudge(
            inference_model=WMLInferenceEngineChat(
                model_name="llama-3-3-70b-instruct",
                max_tokens=5,
                temperature=0.0,
                top_logprobs=5,
                external_client=external_client
            ),
            template="templates.rag_eval.answer_correctness.judge_loose_match_no_context_numeric",
            task="tasks.rag_eval.answer_correctness.binary",
            format=None,
            main_score="answer_correctness_judge",
            prediction_field="answer",
            infer_log_probs=False,
            judge_to_generator_fields_mapping={},
            include_meta_data=False,
        )

        df = pd.DataFrame(
            data=[
                ["In which continent is France?", "France is in Europe." , ["France is a country located in Europe."]],
                ["In which continent is England?", "England is in Europe.", ["England is a country located in Europe."]],
            ],
            columns=["question", "prediction", "ground_truths"]
        )

        results_df, global_scores = evaluate(df, [metric])
        results_df = results_df.round(2)
        instance_scores = results_df.iloc[:, 3]
        self.assertAlmostEqual(list(instance_scores), [0, 0])
