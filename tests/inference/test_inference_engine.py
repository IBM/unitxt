from typing import Any, Dict, List, cast

from unitxt import produce
from unitxt.api import load_dataset
from unitxt.error_utils import UnitxtError
from unitxt.image_operators import EncodeImageToString
from unitxt.inference import (
    HFAutoModelInferenceEngine,
    HFLlavaInferenceEngine,
    HFOptionSelectingInferenceEngine,
    HFPipelineBasedInferenceEngine,
    IbmGenAiInferenceEngine,
    LiteLLMInferenceEngine,
    OptionSelectingByLogProbsInferenceEngine,
    TextGenerationInferenceOutput,
    WMLInferenceEngineChat,
    WMLInferenceEngineGeneration,
)
from unitxt.settings_utils import get_settings
from unitxt.text_utils import print_dict
from unitxt.type_utils import isoftype

from tests.utils import UnitxtInferenceTestCase

settings = get_settings()


class TestInferenceEngine(UnitxtInferenceTestCase):
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
        with self.assertRaises(UnitxtError) as e:
            inference_model.infer(dataset)
        self.assertEqual(
            str(e.exception).strip(),
            f"The instance '{dataset[0]} 'has the following data classification policy "
            f"'{dataset[0]['data_classification_policy']}', however, the artifact "
            f"'{inference_model.get_pretty_print_name()}' is only configured to support the data with "
            f"classification '{inference_model.data_classification_policy}'. To enable this either change "
            f"the 'data_classification_policy' attribute of the artifact, or modify the environment variable "
            f"'UNITXT_DATA_CLASSIFICATION_POLICY' accordingly.\n"
            f"For more information: see https://www.unitxt.ai/en/latest//docs/data_classification_policy.html".strip(),
        )

    def test_llava_inference_engine(self):
        inference_model = HFLlavaInferenceEngine(
            model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=3
        )

        if not settings.use_eager_execution:
            dataset = load_dataset(
                card="cards.doc_vqa.en",
                template="templates.qa.with_context.with_type",
                format="formats.chat_api",
                loader_limit=30,
                split="test",
            )

            predictions = inference_model.infer([dataset[0]])

            self.assertEqual(predictions[0], "The real image")

            prediction = inference_model.infer_log_probs([dataset[1]])[0]

            assert isoftype(prediction, List[Dict[str, Any]])
            self.assertListEqual(
                list(prediction[0].keys()),
                ["text", "logprob", "top_tokens"],
            )

    def test_watsonx_inference(self):
        wml_engine = WMLInferenceEngineGeneration(
            model_name="google/flan-t5-xl",
            data_classification_policy=["public"],
            random_seed=111,
            min_new_tokens=16,
            max_new_tokens=128,
            top_p=0.5,
            top_k=1,
            repetition_penalty=1.5,
            decoding_method="greedy",
        )

        # Loading dataset:
        dataset = load_dataset(
            card="cards.go_emotions.simplified",
            template="templates.classification.multi_label.empty",
            loader_limit=3,
        )
        test_data = dataset["test"]

        # Performing inference:
        predictions = wml_engine.infer(test_data)
        for inp, prediction in zip(test_data, predictions):
            result = {**inp, "prediction": prediction}
            print_dict(result, keys_to_print=["source", "prediction"])

    def test_option_selecting_by_log_prob_inference_engines(self):
        dataset = [
            {
                "source": "hello how are you ",
                "task_data": {"options": ["world", "truck"]},
            },
            {"source": "by ", "task_data": {"options": ["the", "truck"]}},
            # multiple options with the same token prefix
            {
                "source": "I will give you my ",
                "task_data": {
                    "options": [
                        "telephone number",
                        "truck monster",
                        "telephone address",
                    ]
                },
            },
        ]

        genai_engine = IbmGenAiInferenceEngine(
            model_name="mistralai/mixtral-8x7b-instruct-v01"
        )
        watsonx_engine = WMLInferenceEngineGeneration(
            model_name="mistralai/mixtral-8x7b-instruct-v01"
        )

        for engine in [genai_engine, watsonx_engine]:
            dataset = cast(OptionSelectingByLogProbsInferenceEngine, engine).select(
                dataset
            )
            self.assertEqual(dataset[0]["prediction"], "world")
            self.assertEqual(dataset[1]["prediction"], "the")
            self.assertEqual(dataset[2]["prediction"], "telephone number")

    def test_hf_auto_model_inference_engine(self):
        data = load_dataset(
            dataset_query="card=cards.rte,template_card_index=0,loader_limit=20"
        )["test"]

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

    def test_watsonx_inference_with_images(self):
        raw_dataset = load_dataset(
            dataset_query="card=cards.doc_vqa.en,template_card_index=0,loader_limit=30"
        )
        sample = list(raw_dataset["test"])[:2]

        image_encoder = EncodeImageToString()

        inference_engine = WMLInferenceEngineChat(
            model_name="meta-llama/llama-3-2-11b-vision-instruct",
            image_encoder=image_encoder,
            max_tokens=128,
            top_logprobs=3,
        )

        results = inference_engine.infer_log_probs(sample, return_meta_data=True)

        assert isoftype(results, List[TextGenerationInferenceOutput])
        assert results[0].input_tokens == 6541
        assert results[0].stop_reason == "stop"
        assert isoftype(results[0].prediction, List[Dict[str, Any]])

        formatted_dataset = load_dataset(
            card="cards.doc_vqa.en",
            template="templates.qa.with_context.with_type",
            format="formats.chat_api",
            loader_limit=30,
            split="test",
        )
        sample = [formatted_dataset[0]]

        messages = inference_engine.to_messages(sample[0])[0]

        assert isoftype(messages, List[Dict[str, Any]])
        inference_engine.verify_messages(messages)

        inference_engine.top_logprobs = None
        results = inference_engine.infer(sample)

        assert isinstance(results[0], str)

    def test_lite_llm_inference_engine(self):
        from unitxt.logging_utils import set_verbosity

        set_verbosity("debug")
        inference_model = LiteLLMInferenceEngine(
            model="watsonx/meta-llama/llama-3-8b-instruct",
            max_tokens=2,
        )
        recipe = "card=cards.almost_evil,template=templates.qa.open.simple,demos_pool_size=0,num_demos=0,format=formats.chat_api"
        instances = [
            {
                "question": "How many days there are in a week? answer just the number in digits",
                "answers": ["7"],
            },
            {
                "question": "If a ate an apple in the morning, and one in the evening, how many apples did I eat? answer just the number in digits",
                "answers": ["2"],
            },
        ]
        total_tests = 5
        instances = (instances * (total_tests // len(instances)))[:total_tests]
        dataset = produce(instances, recipe)

        predictions = inference_model.infer(dataset)

        targets = ["7", "2"]
        targets = (targets * (total_tests // len(targets)))[:total_tests]
        self.assertListEqual(predictions, targets)

    def test_log_prob_scoring_inference_engine(self):
        engine = HFOptionSelectingInferenceEngine(model_name="gpt2", batch_size=1)

        log_probs = engine.get_log_probs(["hello world", "by universe"])

        self.assertAlmostEqual(log_probs[0], -8.58, places=2)
        self.assertAlmostEqual(log_probs[1], -10.98, places=2)

    def test_option_selecting_inference_engine(self):
        dataset = [
            {"source": "hello ", "task_data": {"options": ["world", "truck"]}},
            {"source": "by ", "task_data": {"options": ["the", "truck"]}},
        ]

        engine = HFOptionSelectingInferenceEngine(model_name="gpt2", batch_size=1)
        predictions = engine.infer(dataset)

        self.assertEqual(predictions[0], "world")
        self.assertEqual(predictions[1], "the")

    def test_option_selecting_inference_engine_chat_api(self):
        dataset = [
            {
                "source": [{"role": "user", "content": "hi you!"}],
                "task_data": {"options": ["hello friend", "hello truck"]},
            },
            {
                "source": [{"role": "user", "content": "black or white?"}],
                "task_data": {"options": ["white.", "white truck"]},
            },
        ]

        engine = HFOptionSelectingInferenceEngine(
            model_name="Qwen/Qwen2.5-0.5B-Instruct", batch_size=1
        )
        predictions = engine.infer(dataset)

        self.assertEqual(predictions[0], "hello friend")
        self.assertEqual(predictions[1], "white.")
