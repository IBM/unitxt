import json
import unittest
from typing import Any, Dict, List

from datasets.arrow_dataset import Dataset
from unitxt import produce
from unitxt.api import load_dataset
from unitxt.inference import (
    HFAutoModelInferenceEngine,
    HFLlavaInferenceEngine,
    HFPipelineBasedInferenceEngine,
    MockInferenceEngine,
    TextGenerationInferenceOutput,
    WMLInferenceEngine,
    mock_logprobs_default_value_factory,
)
from unitxt.metrics import Accuracy, Metric
from unitxt.operator import MissingRequirementsError
from unitxt.settings_utils import get_settings
from unitxt.standard import StandardRecipe
from unitxt.test_utils.metrics import apply_metric
from unitxt.type_utils import isoftype

from tests.utils import UnitxtTestCase

settings = get_settings()


class TestInferenceEngine(UnitxtTestCase):
    @staticmethod
    def prepare_test_data() -> Dataset:
        return load_dataset(
            dataset_query="card=cards.rte,template_card_index=0,loader_limit=20"
        )["test"]

    @staticmethod
    def get_test_references(task_data: List[str]) -> List[List[List[str]]]:
        return [[[json.loads(sample)["label"]]] for sample in task_data]

    @staticmethod
    def process_predictions(predictions: List[str]) -> List[List[str]]:
        return [[prediction] for prediction in predictions]

    @staticmethod
    def calculate_metric_global_score(
        metric: Metric, preds: List[List[str]], refs: List[List[List[str]]]
    ) -> float:
        return apply_metric(metric=metric, predictions=preds, references=refs)[0][
            "score"
        ]["global"]["score"]

    def test_pipeline_based_inference_engine(self):
        inference_model = HFPipelineBasedInferenceEngine(
            model_name="google/flan-t5-small", max_new_tokens=32
        )
        assert inference_model._is_loaded()

        recipe = "card=cards.almost_evil,template=templates.qa.open.simple,demos_pool_size=0,num_demos=0"
        instances = [
            {"question": "How many days there are in a week", "answers": ["7"]},
            {
                "question": "If a ate an apple in the morning, and one in the evening, how many apples did I eat?",
                "answers": ["2"],
            },
        ]
        dataset = produce(instances, recipe)

        predictions = inference_model.infer(dataset)

        targets = ["365", "1"]
        self.assertListEqual(predictions, targets)

    def test_pipeline_based_inference_engine_lzay_load(self):
        inference_model = HFPipelineBasedInferenceEngine(
            model_name="google/flan-t5-small", max_new_tokens=32, lazy_load=True
        )
        assert not inference_model._is_loaded()
        recipe = "card=cards.almost_evil,template=templates.qa.open.simple,demos_pool_size=0,num_demos=0"
        instances = [
            {"question": "How many days there are in a week", "answers": ["7"]},
            {
                "question": "If a ate an apple in the morning, and one in the evening, how many apples did I eat?",
                "answers": ["2"],
            },
        ]
        dataset = produce(instances, recipe)

        predictions = inference_model.infer(dataset)

        targets = ["365", "1"]
        self.assertListEqual(predictions, targets)

    def test_dataset_verification_inference_engine(self):
        inference_model = HFPipelineBasedInferenceEngine(
            model_name="google/flan-t5-small",
            max_new_tokens=32,
            data_classification_policy=["public"],
        )
        dataset = [{"source": "", "data_classification_policy": ["pii"]}]
        with self.assertRaises(ValueError) as e:
            inference_model.infer(dataset)
        self.assertEqual(
            str(e.exception),
            f"The instance '{dataset[0]} 'has the following data classification policy "
            f"'{dataset[0]['data_classification_policy']}', however, the artifact "
            f"'{inference_model.get_pretty_print_name()}' is only configured to support the "
            f"data with classification '{inference_model.data_classification_policy}'. To "
            f"enable this either change the 'data_classification_policy' attribute of the "
            f"artifact, or modify the environment variable 'UNITXT_DATA_CLASSIFICATION_POLICY' "
            f"accordingly.",
        )

    def test_llava_inference_engine(self):
        inference_model = HFLlavaInferenceEngine(
            model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=3
        )

        if not settings.use_eager_execution:
            dataset = StandardRecipe(
                card="cards.doc_vqa.en",
                template="templates.qa.with_context.with_type",
                format="formats.models.llava_interleave",
                loader_limit=30,
            )()

            test_dataset = [dataset["test"].peek()]

            predictions = inference_model.infer(test_dataset)

            self.assertEqual(predictions[0], "The real image")

            prediction = inference_model.infer_log_probs(test_dataset)[0]

            assert isoftype(prediction, List[Dict[str, Any]])
            self.assertListEqual(
                list(prediction[0].keys()),
                ["text", "logprob", "top_tokens"],
            )

    def test_wml_inference_engine(self):
        try:
            inference_engine = WMLInferenceEngine(
                model_name="google/flan-t5-xl",
                random_seed=123,
                concurrency_limit=5,
            )

            dataset = self.prepare_test_data().take(5)

            results = inference_engine._infer(dataset)

            assert len(results) == len(dataset["source"])
            assert results[0] == "entailment"

            references = self.get_test_references(dataset["task_data"])
            processed_results = self.process_predictions(results)
            acc = Accuracy()
            score = self.calculate_metric_global_score(
                acc, processed_results, references
            )
            assert score == 0.4
        except (MissingRequirementsError, ModuleNotFoundError):
            # In such case, the test is omitted as not every user may
            # need to use this package.
            pass

    def test_wml_inference_engine_with_logprobs_and_meta_results(self):
        try:
            inference_engine = WMLInferenceEngine(
                model_name="google/flan-t5-xl",
                random_seed=123,
                concurrency_limit=5,
            )

            dataset = self.prepare_test_data().take(5)

            results = inference_engine._infer_log_probs(dataset, return_meta_data=True)
            sample = results[0]

            assert len(results) == len(dataset["source"])
            assert isinstance(sample, TextGenerationInferenceOutput)
            assert sample.seed == 123
            assert sample.stop_reason == "eos_token"
            assert len(sample.prediction) == 5
            self.assertListEqual(
                list(sample.prediction[0].keys()),
                ["text", "logprob", "top_tokens"],
            )
            self.assertListEqual(
                [token["text"] for token in sample.prediction],
                ["▁", "en", "tail", "ment", "</s>"],
            )
        except (MissingRequirementsError, ModuleNotFoundError):
            # In such case, the test is omitted as not every user may
            # need to use this package.
            pass

    def test_mock_inference_engine(self):
        dataset = self.prepare_test_data()

        inference_engine = MockInferenceEngine(model_name="model")

        assert inference_engine.get_engine_id() == "model_mock"
        assert inference_engine.get_model_details() == {}

        results = inference_engine.infer(dataset)

        assert len(results) == len(dataset)
        assert results[0] == inference_engine.default_inference_value

        results = inference_engine.infer_log_probs(dataset, return_meta_data=True)
        sample = results[0]

        assert len(results) == len(dataset)
        assert isinstance(sample, TextGenerationInferenceOutput)
        assert sample.prediction == mock_logprobs_default_value_factory()
        assert sample.seed == 111
        assert sample.stop_reason == ""
        assert sample.output_tokens == len(mock_logprobs_default_value_factory())
        assert sample.input_tokens == len(dataset[0]["source"])
        assert sample.input_text == dataset[0]["source"]

    def test_hf_auto_model_inference_engine(self):
        data = self.prepare_test_data()

        engine = HFAutoModelInferenceEngine(
            model_name="google/flan-t5-small",
            max_new_tokens=16,
            repetition_penalty=1.5,
            top_k=5,
            data_classification_policy=["public"],
        )

        assert engine.get_engine_id() == "flan_t5_small_hf_auto_model"
        assert engine.repetition_penalty == 1.5

        results = engine.infer_log_probs(data, return_meta_data=True)
        sample = results[0]
        prediction = sample.prediction

        assert len(results) == len(data)
        assert isinstance(sample, TextGenerationInferenceOutput)
        assert sample.output_tokens == 5
        assert isoftype(prediction, List[Dict[str, Any]])
        self.assertListEqual(
            list(prediction[0].keys()),
            ["text", "logprob", "top_tokens"],
        )
        assert isinstance(prediction[0]["text"], str)
        assert isinstance(prediction[0]["logprob"], float)

        results = engine.infer(data)

        assert isoftype(results, List[str])
        assert results[0] == "entailment"


if __name__ == "__main__":
    unittest.main()
