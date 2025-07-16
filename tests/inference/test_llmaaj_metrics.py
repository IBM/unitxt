import os

import pandas as pd
import unitxt
from unitxt.api import create_dataset
from unitxt.api import evaluate as evaluate_api
from unitxt.artifact import fetch_artifact
from unitxt.blocks import Task
from unitxt.error_utils import UnitxtError
from unitxt.eval_utils import evaluate
from unitxt.inference import CrossProviderInferenceEngine, WMLInferenceEngineChat
from unitxt.llm_as_judge import LLMJudgeDirect
from unitxt.llm_as_judge_constants import CriteriaWithOptions

from tests.utils import UnitxtInferenceTestCase


class TestLLMaaJFromTemplateMetrics(UnitxtInferenceTestCase):
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

    def test_evaluate___answer_correctness__external_client__fetch_artifact(self):
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
            artifact_rep="metrics.rag.external_rag.answer_correctness.llama_3_3_70b_instruct_watsonx_judge",
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

    def test_evaluate___faithfulness__external_client__fetch_artifact(self):
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
            artifact_rep="metrics.rag.external_rag.faithfulness.llama_3_3_70b_instruct_watsonx_judge",
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
            columns=["question", "prediction", "contexts"],
        )

        results_df, global_scores = evaluate(df, [metric])
        results_df = results_df.round(2)
        instance_scores = results_df.iloc[:, 3]
        self.assertListEqual(list(instance_scores), [1.0, 1.0])


class TestLLMaaJMetrics(UnitxtInferenceTestCase):
    def test_llm_judge_prediction_and_context_fields_config(self):
        ### Scenarios to test:

        # Scenario 1: predictions are taken from the evaluate param and context is taken from the context_fields LLMJudge param

        # Scenario 2: both predictions and context names are taken from the criteria fields

        # Scenario 3: context fields is missing both from the LLMJudge param (None or []) and from the criteria.context_fields fields (None or [])

        # Scenario 4: predictions and context names are expressed both as in Scenario 1 and as in Scenario 2, but Scenario 1 (evaluate param for preediction and LLMJudge param for context) takes prevalence

        unitxt.settings.mock_inference_mode = True

        # Scenario 1:
        with self.subTest(
            "Scenario 1: predictions are taken from the evaluate param and context is taken from the context_fields LLMJudge param"
        ):
            criteria_without_prediction_context_fields = CriteriaWithOptions.from_obj(
                {
                    "name": "testing_criteria",
                    "description": "description",
                    "options": [
                        {
                            "name": "Option 1",
                            "description": "",
                        },
                        {
                            "name": "Option 2",
                            "description": "",
                        },
                    ],
                    "option_map": {"Option 1": 1.0, "Option 2": 0.0},
                }
            )

            metric = LLMJudgeDirect(
                inference_engine=CrossProviderInferenceEngine(
                    model="llama-3-3-70b-instruct",
                    max_tokens=1024,
                    data_classification_policy=["public"],
                ),
                include_prompts_in_result=True,
                criteria=criteria_without_prediction_context_fields,
                context_fields=["question"],
            )

            data = [
                {"question": "question taken from LLMJudge context_fields param"},
            ]

            dataset = create_dataset(
                task="tasks.qa.open", test_set=data, metrics=[metric], split="test"
            )

            predictions = [
                "prediction taken from evaluate()'s prediction param",
            ]

            results = evaluate_api(predictions=predictions, data=dataset)
            assessment_prompt = results[0]["score"]["instance"][
                "testing_criteria_prompts"
            ]["assessment"][0]["content"]
            self.assertIn(
                "question taken from LLMJudge context_fields param", assessment_prompt
            )
            self.assertIn(
                "prediction taken from evaluate()'s prediction param", assessment_prompt
            )

        # Scenario 2:
        with self.subTest(
            "Scenario 2:  both predictions and context names are taken from the criteria fields"
        ):
            criteria_with_prediction_context_fields = CriteriaWithOptions.from_obj(
                {
                    "name": "testing_criteria",
                    "description": "description",
                    "options": [
                        {
                            "name": "Option 1",
                            "description": "",
                        },
                        {
                            "name": "Option 2",
                            "description": "",
                        },
                    ],
                    "option_map": {"Option 1": 1.0, "Option 2": 0.0},
                    "prediction_field": "answer",
                    "context_fields": {"question in prompt": "question_in_dataset"},
                }
            )

            data = [
                {
                    "question_in_dataset": "question taken from criteria context_fields param",
                    "answer": "prediction taken from criteria prediction_field param",
                },
            ]

            metric = LLMJudgeDirect(
                inference_engine=CrossProviderInferenceEngine(
                    model="llama-3-3-70b-instruct",
                    max_tokens=1024,
                    data_classification_policy=["public"],
                ),
                include_prompts_in_result=True,
                criteria=criteria_with_prediction_context_fields,
                context_fields=None,
            )

            dataset = create_dataset(
                task=Task(
                    input_fields={"question_in_dataset": str, "answer": str},
                    reference_fields={},
                    prediction_type=str,
                    metrics=[metric],
                ),
                test_set=data,
                metrics=[metric],
                split="test",
                template="templates.empty",
            )

            results = evaluate_api(data=dataset)

            assessment_prompt = results[0]["score"]["instance"][
                "testing_criteria_prompts"
            ]["assessment"][0]["content"]
            self.assertIn(
                "question in prompt: question taken from criteria context_fields param",
                assessment_prompt,
            )
            self.assertIn(
                "prediction taken from criteria prediction_field param",
                assessment_prompt,
            )

        # Scenario 3:
        with self.subTest(
            "Scenario 3: context fields is missing both from the LLMJudge param (None or []) and from the criteria.context_fields fields (None or [])"
        ):
            criteria_with_prediction_context_fields = CriteriaWithOptions.from_obj(
                {
                    "name": "testing_criteria",
                    "description": "description",
                    "options": [
                        {
                            "name": "Option 1",
                            "description": "",
                        },
                        {
                            "name": "Option 2",
                            "description": "",
                        },
                    ],
                    "option_map": {"Option 1": 1.0, "Option 2": 0.0},
                }
            )

            data = [
                {
                    "question": "question taken from LLMJudge context_fields param",
                    "model_response": "On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.",
                },
            ]

            metric = LLMJudgeDirect(
                inference_engine=CrossProviderInferenceEngine(
                    model="llama-3-3-70b-instruct",
                    max_tokens=1024,
                    data_classification_policy=["public"],
                ),
                include_prompts_in_result=True,
                criteria=criteria_with_prediction_context_fields,
                context_fields=None,
            )

            dataset = create_dataset(
                task="tasks.qa.open", test_set=data, metrics=[metric], split="test"
            )
            with self.assertRaises(UnitxtError) as context:
                evaluate_api(data=dataset)

            self.assertEqual(
                str(context.exception.original_error),
                "You must set either the predictions in the evaluate() call or specify the prediction field name to be taken from the task_data using the `Criteria`'s prediction_field field.",
            )

        # Scenario 4:
        with self.subTest(
            "Scenario 4: predictions and context names are expressed both as in Scenario 1 and as in Scenario 2, but Scenario 1 (evaluate param for preediction and LLMJudge param for context) takes prevalence"
        ):
            criteria_with_prediction_context_fields = CriteriaWithOptions.from_obj(
                {
                    "name": "testing_criteria",
                    "description": "description",
                    "options": [
                        {
                            "name": "Option 1",
                            "description": "",
                        },
                        {
                            "name": "Option 2",
                            "description": "",
                        },
                    ],
                    "option_map": {"Option 1": 1.0, "Option 2": 0.0},
                    "prediction_field": "answer",
                    "context_fields": ["alternative_question"],
                }
            )

            data = [
                {
                    "question": "question taken from LLMJudge context_fields param",
                    "alternative_question": "question taken from criteria context_fields param",
                    "answer": "prediction taken from criteria prediction_field param",
                },
            ]

            metric = LLMJudgeDirect(
                inference_engine=CrossProviderInferenceEngine(
                    model="llama-3-3-70b-instruct",
                    max_tokens=1024,
                    data_classification_policy=["public"],
                ),
                include_prompts_in_result=True,
                criteria=criteria_with_prediction_context_fields,
                context_fields=["question"],
            )

            dataset = create_dataset(
                task=Task(
                    input_fields={
                        "question": str,
                        "alternative_question": str,
                        "answer": str,
                    },
                    reference_fields={},
                    prediction_type=str,
                    metrics=[metric],
                ),
                test_set=data,
                metrics=[metric],
                split="test",
                template="templates.empty",
            )

            predictions = [
                "prediction taken from evaluate()'s prediction param",
            ]

            results = evaluate_api(predictions=predictions, data=dataset)
            assessment_prompt = results[0]["score"]["instance"][
                "testing_criteria_prompts"
            ]["assessment"][0]["content"]
            self.assertIn(
                "question: question taken from LLMJudge context_fields param",
                assessment_prompt,
            )
            self.assertNotIn(
                "question taken from criteria context_fields param", assessment_prompt
            )
            self.assertIn(
                "prediction taken from evaluate()'s prediction param", assessment_prompt
            )
