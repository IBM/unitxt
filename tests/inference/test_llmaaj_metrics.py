import os

import pandas as pd
from unitxt.artifact import fetch_artifact
from unitxt.eval_utils import evaluate
from unitxt.inference import WMLInferenceEngineChat

from tests.utils import UnitxtInferenceTestCase


class TestLLMaaJMetrics(UnitxtInferenceTestCase):
    def test_evaluate_with_llmaaj_and_external_client(self):
        from ibm_watsonx_ai.client import APIClient, Credentials
        from unitxt.inference import WMLInferenceEngineChat
        from unitxt.llm_as_judge_from_template import TaskBasedLLMasJudge

        external_client = APIClient(
            credentials=Credentials(
                api_key=os.environ.get("WML_APIKEY"), url=os.environ.get("WML_URL")
            ),
            project_id=os.environ.get("WML_PROJECT_ID"),
        )

        metric = TaskBasedLLMasJudge(
            inference_model=WMLInferenceEngineChat(
                model_name="meta-llama/llama-3-3-70b-instruct",
                max_tokens=5,
                temperature=0.0,
                external_client=external_client,
            ),
            template="templates.rag_eval.answer_correctness.judge_loose_match_no_context_numeric",
            task="tasks.rag_eval.answer_correctness.binary",
            format=None,
            main_score="answer_correctness_judge",
            prediction_field="answer",
            include_meta_data=False,
        )

        df = pd.DataFrame(
            data=[
                [
                    "In which continent is France?",
                    "France is in Europe.",
                    ["France is a country located in Europe."],
                ],
                [
                    "In which continent is England?",
                    "England is in Europe.",
                    ["England is a country located in Europe."],
                ],
            ],
            columns=["question", "prediction", "ground_truths"],
        )

        results_df, global_scores = evaluate(df, [metric])
        results_df = results_df.round(2)
        instance_scores = results_df.iloc[:, 3]
        self.assertListEqual(list(instance_scores), [1.0, 1.0])

    def test_evaluate_with_llmaaj_and_external_client__with_fetch_artifact(self):
        from ibm_watsonx_ai.client import APIClient, Credentials

        external_client = APIClient(
            credentials=Credentials(
                api_key=os.environ.get("WML_APIKEY"), url=os.environ.get("WML_URL")
            ),
            project_id=os.environ.get("WML_PROJECT_ID"),
        )

        inference_model = WMLInferenceEngineChat(
            model_name="meta-llama/llama-3-3-70b-instruct",
            max_tokens=5,
            temperature=0.0,
            external_client=external_client,
        )

        metric, _ = fetch_artifact(
            artifact_rep="metrics.rag.answer_correctness.llama_3_1_70b_instruct_wml_q_a_gt_loose_numeric",
            overwrite_kwargs={
                "inference_model": inference_model,
                "include_meta_data": False,
            },
        )

        df = pd.DataFrame(
            data=[
                [
                    "In which continent is France?",
                    "France is in Europe.",
                    ["France is a country located in Europe."],
                ],
                [
                    "In which continent is England?",
                    "England is in Europe.",
                    ["England is a country located in Europe."],
                ],
            ],
            columns=["question", "prediction", "ground_truths"],
        )

        results_df, global_scores = evaluate(df, [metric])
        results_df = results_df.round(2)
        instance_scores = results_df.iloc[:, 3]
        self.assertListEqual(list(instance_scores), [1.0, 1.0])
