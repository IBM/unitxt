import os
import random
import shutil
import time
from functools import lru_cache
from typing import Any, Dict, List, cast

import unitxt
from unitxt import create_dataset
from unitxt.api import load_dataset
from unitxt.error_utils import UnitxtError
from unitxt.inference import (
    HFAutoModelInferenceEngine,
    HFGraniteSpeechInferenceEngine,
    HFLlavaInferenceEngine,
    HFOptionSelectingInferenceEngine,
    HFPipelineBasedInferenceEngine,
    LiteLLMInferenceEngine,
    OllamaInferenceEngine,
    OptionSelectingByLogProbsInferenceEngine,
    RITSInferenceEngine,
    TextGenerationInferenceOutput,
    WMLInferenceEngineChat,
    WMLInferenceEngineGeneration,
)
from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_settings
from unitxt.type_utils import isoftype

from tests.utils import UnitxtInferenceTestCase

logger = get_logger()
settings = get_settings()

local_decoder_model = "HuggingFaceTB/SmolLM2-135M-Instruct"  # pragma: allowlist secret


@lru_cache
def get_image_dataset(format=None):
    import numpy as np
    from PIL import Image

    random_image = Image.fromarray(
        np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    )

    data = [
        {
            "context": {"image": random_image, "format": "JPEG"},
            "context_type": "image",
            "question": "What is the capital of Texas?",
            "answers": ["Austin"],
        },
        {
            "context": {"image": random_image, "format": "JPEG"},
            "context_type": "image",
            "question": "What is the color of the sky?",
            "answers": ["Blue"],
        },
    ]

    return create_dataset(
        task="tasks.qa.with_context",
        format=format,
        test_set=data,
        split="test",
        data_classification_policy=["public"],
    )


@lru_cache
def get_audio_dataset(format=None):
    import numpy as np

    # Generate synthetic audio data (1 second of 16kHz audio)
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)

    # Generate synthetic audio (simple sine wave)
    frequency = 440  # A4 note
    time_values = np.linspace(0, duration, num_samples)
    audio_data = np.sin(2 * np.pi * frequency * time_values).astype(np.float32)

    data = [
        {
            "context": {"audio": {"array": audio_data, "sampling_rate": sample_rate}},
            "context_type": "audio",
            "question": "What is the main topic of this audio?",
            "answers": ["Music"],
        },
        {
            "context": {"audio": {"array": audio_data, "sampling_rate": sample_rate}},
            "context_type": "audio",
            "question": "Describe the audio content",
            "answers": ["Tone"],
        },
    ]

    return create_dataset(
        task="tasks.qa.with_context",
        format=format,
        test_set=data,
        split="test",
        data_classification_policy=["public"],
    )


@lru_cache
def get_text_dataset(format=None):
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

    return create_dataset(
        task="tasks.qa.open",
        format=format,
        test_set=instances,
        split="test",
        template="templates.qa.open.simple",
        data_classification_policy=["public"],
    )


class TestInferenceEngine(UnitxtInferenceTestCase):
    def test_pipeline_based_inference_engine(self):
        model = HFPipelineBasedInferenceEngine(
            model_name=local_decoder_model,  # pragma: allowlist secret
            max_new_tokens=2,
        )

        dataset = get_text_dataset()

        predictions = model(dataset)

        self.assertListEqual(list(predictions), ["7\n", "12"])

    def test_pipeline_based_inference_engine_lazy_load(self):
        model = HFPipelineBasedInferenceEngine(
            model_name=local_decoder_model,  # pragma: allowlist secret
            max_new_tokens=2,
            lazy_load=True,
        )
        dataset = get_text_dataset()

        predictions = model(dataset)

        self.assertListEqual(list(predictions), ["7\n", "12"])

    def test_dataset_verification_inference_engine(self):
        inference_model = HFPipelineBasedInferenceEngine(
            model_name=local_decoder_model,  # pragma: allowlist secret
            max_new_tokens=2,
            lazy_load=True,
            data_classification_policy=["public"],
        )
        dataset = [{"source": "", "data_classification_policy": ["pii"]}]
        with self.assertRaises(UnitxtError) as e:
            inference_model.infer(dataset)
        self.assertIn(
            f"The instance '{dataset[0]} 'has the following data classification policy "
            f"'{dataset[0]['data_classification_policy']}', however, the artifact "
            f"'{inference_model.get_pretty_print_name()}' is only configured to support the data with "
            f"classification '{inference_model.data_classification_policy}'. To enable this either change "
            f"the 'data_classification_policy' attribute of the artifact, or modify the environment variable "
            f"'UNITXT_DATA_CLASSIFICATION_POLICY' accordingly.\n"
            f"For more information: see https://www.unitxt.ai/en/latest//docs/data_classification_policy.html".strip(),
            str(e.exception).strip(),
        )

    def test_llava_inference_engine(self):
        model = HFLlavaInferenceEngine(
            model_name="llava-hf/llava-interleave-qwen-0.5b-hf",
            max_new_tokens=3,
            temperature=0.0,
        )

        dataset = get_image_dataset(format="formats.chat_api")

        predictions = model.infer(dataset)

        self.assertListEqual(predictions, ["Austin", "Blue"])

        prediction = model.infer_log_probs(dataset)

        assert isoftype(prediction, List[List[Dict[str, Any]]])
        self.assertListEqual(
            list(prediction[0][0].keys()),
            ["text", "logprob", "top_tokens"],
        )

    def test_granite_speech_inference_engine(self):
        model = HFGraniteSpeechInferenceEngine(
            model_name="ibm-granite/granite-speech-3.3-2b",
            max_new_tokens=10,
            temperature=0.0,
        )

        # Test with chat API format
        dataset = get_audio_dataset(format="formats.chat_api")

        predictions = model.infer(dataset)

        # Check that we get predictions for both instances
        self.assertEqual(len(predictions), 2)
        self.assertIsInstance(predictions[0], str)
        self.assertIsInstance(predictions[1], str)

        # Test log probabilities inference
        prediction = model.infer_log_probs(dataset)

        assert isoftype(prediction, List[List[Dict[str, Any]]])
        self.assertListEqual(
            list(prediction[0][0].keys()),
            ["text", "logprob", "top_tokens"],
        )

    def test_watsonx_inference(self):
        model = WMLInferenceEngineGeneration(
            model_name="google/flan-t5-xl",
            data_classification_policy=["public"],
            random_seed=111,
            min_new_tokens=1,
            max_new_tokens=3,
            top_p=0.5,
            top_k=1,
            repetition_penalty=1.5,
            decoding_method="greedy",
        )

        dataset = get_text_dataset()

        predictions = model(dataset)

        self.assertListEqual(predictions, ["7", "2"])

    def test_watsonx_chat_inference(self):
        model = WMLInferenceEngineChat(
            model_name="ibm/granite-3-8b-instruct",
            data_classification_policy=["public"],
            temperature=0,
        )

        dataset = get_text_dataset()

        predictions = model(dataset)

        self.assertListEqual(predictions, ["7", "2"])

    def test_watsonx_inference_with_external_client(self):
        from ibm_watsonx_ai.client import APIClient, Credentials

        model = WMLInferenceEngineGeneration(
            model_name="google/flan-t5-xl",
            data_classification_policy=["public"],
            random_seed=111,
            min_new_tokens=1,
            max_new_tokens=3,
            top_p=0.5,
            top_k=1,
            repetition_penalty=1.5,
            decoding_method="greedy",
            external_client=APIClient(
                credentials=Credentials(
                    api_key=os.environ.get("WML_APIKEY"), url=os.environ.get("WML_URL")
                ),
                project_id=os.environ.get("WML_PROJECT_ID"),
            ),
        )

        dataset = get_text_dataset()

        predictions = model(dataset)

        self.assertListEqual(predictions, ["7", "2"])

    def test_rits_inference(self):
        import os

        if os.environ.get("RITS_API_KEY") is None:
            logger.warning(
                "Skipping test_rits_inference because RITS_API_KEY not defined"
            )
            return

        model = RITSInferenceEngine(
            model_name="microsoft/phi-4",
            max_tokens=128,
        )

        dataset = get_text_dataset()

        predictions = model(dataset)

        self.assertListEqual(predictions, ["7", "2"])

    def test_rits_byom_inference(self):
        import os

        if os.environ.get("RITS_BYOM_IS_UP") is None:
            logger.warning(
                "Skipping RITS_BYOM_IS_UP not defined. "
                "In order to start RITS BYOM model please use 'gb build init model_to_rits --from-template ModelToRITS'"
                "and start gb."
            )
            return

        model = RITSInferenceEngine(
            model_name="byom-gb-iqk-lora/ibm-granite/granite-3.1-8b-instruct",
            max_tokens=128,
        )

        dataset = get_text_dataset()

        predictions = model(dataset)

        self.assertListEqual(predictions, ["7", "2"])

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

        watsonx_engine = WMLInferenceEngineGeneration(
            model_name="mistralai/mistral-small-3-1-24b-instruct-2503"
        )

        for engine in [watsonx_engine]:
            dataset = cast(OptionSelectingByLogProbsInferenceEngine, engine).select(
                dataset
            )
            self.assertEqual(dataset[0]["prediction"], "world")
            self.assertEqual(dataset[1]["prediction"], "the")
            self.assertEqual(dataset[2]["prediction"], "telephone number")

    def test_hf_auto_model_inference_engine_batching(self):
        model = HFAutoModelInferenceEngine(
            model_name=local_decoder_model,  # pragma: allowlist secret
            max_new_tokens=2,
            batch_size=2,
            data_classification_policy=["public"],
        )

        dataset = get_text_dataset()

        predictions = list(model(dataset))

        self.assertListEqual(predictions, ["7\n", "12"])

    def test_hf_auto_model_inference_engine(self):
        data = get_text_dataset()
        engine = HFAutoModelInferenceEngine(
            model_name="google/flan-t5-small",
            max_new_tokens=16,
            repetition_penalty=1.5,
            top_k=5,
            data_classification_policy=["public"],
        )

        self.assertEqual(engine.get_engine_id(), "flan_t5_small_hf_auto_model")
        self.assertEqual(engine.repetition_penalty, 1.5)

        results = engine.infer_log_probs(data, return_meta_data=True)
        sample = results[0]
        prediction = sample.prediction

        self.assertEqual(engine.repetition_penalty, 1.5)
        self.assertEqual(len(results), len(data))
        self.assertIsInstance(sample, TextGenerationInferenceOutput)
        self.assertEqual(sample.output_tokens, 3)
        self.assertTrue(isoftype(prediction, List[Dict[str, Any]]))
        self.assertListEqual(
            list(prediction[0].keys()),
            ["text", "logprob", "top_tokens"],
        )
        self.assertIsInstance(prediction[0]["text"], str)
        self.assertIsInstance(prediction[0]["logprob"], float)
        self.assertEqual(sample.generated_text, "365")
        results = engine.infer(data)

        self.assertTrue(isoftype(results, List[str]))
        self.assertEqual(results[0], "365")

    def test_watsonx_inference_with_images(self):
        dataset = get_image_dataset()

        inference_engine = WMLInferenceEngineChat(
            model_name="meta-llama/llama-3-2-11b-vision-instruct",
            max_tokens=128,
            top_logprobs=3,
            temperature=0.0,
        )

        results = inference_engine.infer_log_probs(
            dataset.select([0]), return_meta_data=True
        )
        self.assertEqual(results[0].generated_text, "The capital of Texas is Austin.")
        self.assertTrue(isoftype(results, List[TextGenerationInferenceOutput]))
        self.assertEqual(results[0].stop_reason, "stop")
        self.assertTrue(isoftype(results[0].prediction, List[Dict[str, Any]]))

        dataset = get_image_dataset(format="formats.chat_api")

        inference_engine = WMLInferenceEngineChat(
            model_name="meta-llama/llama-3-2-11b-vision-instruct",
            max_tokens=128,
        )

        results = inference_engine.infer(dataset.select([0]))

        self.assertIsInstance(results[0], str)

    def test_lite_llm_inference_engine(self):
        model = LiteLLMInferenceEngine(
            model="watsonx/meta-llama/llama-3-3-70b-instruct",
            max_tokens=2,
            temperature=0,
            top_p=1,
            seed=42,
        )

        dataset = get_text_dataset(format="formats.chat_api")
        predictions = model(dataset)

        self.assertListEqual(predictions, ["7", "2"])

    def test_lite_llm_inference_engine_without_task_data_not_failing(self):
        LiteLLMInferenceEngine(
            model="watsonx/meta-llama/llama-3-3-70b-instruct",
            max_tokens=2,
            temperature=0,
            top_p=1,
            seed=42,
        ).infer([{"source": "say hello."}])

    def test_log_prob_scoring_inference_engine(self):
        engine = HFOptionSelectingInferenceEngine(
            model_name=local_decoder_model,  # pragma: allowlist secret
            batch_size=1,
        )

        log_probs = engine.get_log_probs(["hello world", "by universe"])

        self.assertAlmostEqual(log_probs[0], -9.77, places=2)
        self.assertAlmostEqual(log_probs[1], -11.92, places=2)

    def test_option_selecting_inference_engine(self):
        dataset = [
            {"source": "hello ", "task_data": {"options": ["world", "truck"]}},
            {"source": "by ", "task_data": {"options": ["the", "truck"]}},
        ]

        engine = HFOptionSelectingInferenceEngine(
            model_name=local_decoder_model, batch_size=1
        )
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
            model_name=local_decoder_model, batch_size=1
        )
        predictions = engine.infer(dataset)

        self.assertEqual(predictions[0], "hello friend")
        self.assertEqual(predictions[1], "white.")

    def test_hugginface_pipeline_inference_engine_chat_api(self):
        from transformers import set_seed

        dataset = [
            {
                "source": [{"role": "user", "content": "hi you!"}],
            },
            {
                "source": [{"role": "user", "content": "black or white?"}],
            },
        ]

        set_seed(0, deterministic=True)

        engine = HFPipelineBasedInferenceEngine(
            model_name=local_decoder_model,
            max_new_tokens=1,
            top_k=1,
        )
        predictions = engine.infer(dataset)

        self.assertEqual(predictions[0], "hi")
        self.assertEqual(predictions[1], "I")

    def test_ollama_inference_engine(self):
        dataset = [
            {"source": "Answer in one word only. What is the capital of Canada"},
        ]

        engine = OllamaInferenceEngine(model="llama3.2:1b", temperature=0.0)
        predictions = engine.infer(dataset)

        self.assertTrue("Ottawa" in predictions[0], predictions[0])

    def test_cache(self):
        unitxt.settings.allow_unverified_code = True
        if os.path.exists(unitxt.settings.inference_engine_cache_path):
            shutil.rmtree(unitxt.settings.inference_engine_cache_path)

        model_name = local_decoder_model  # pragma: allowlist secret

        dataset = load_dataset(
            card="cards.openbook_qa",
            split="test",
            # format="formats.chat_api",
            loader_limit=20,
        )
        inference_model = HFPipelineBasedInferenceEngine(
            model_name=model_name,
            max_new_tokens=32,
            temperature=0,
            top_p=1,
            use_cache=False,
            device="cpu",
        )
        start_time = time.time()
        predictions_without_cache = inference_model.infer(dataset)
        inference_without_cache_time = time.time() - start_time
        # Set seed for reproducibility
        inference_model = HFPipelineBasedInferenceEngine(
            model_name=model_name,
            max_new_tokens=32,
            temperature=0,
            top_p=1,
            use_cache=True,
            cache_batch_size=5,
            device="cpu",
        )
        start_time = time.time()
        predictions_with_cache = inference_model.infer(dataset)
        inference_with_cache_time = time.time() - start_time

        self.assertEqual(len(predictions_without_cache), len(predictions_with_cache))
        for p1, p2 in zip(predictions_without_cache, predictions_with_cache):
            self.assertEqual(p1, p2)

        logger.info(
            f"Time of inference without cache: {inference_without_cache_time}, "
            f"with cache (cache is empty): {inference_with_cache_time}"
        )

        start_time = time.time()
        predictions_with_cache = inference_model.infer(dataset)
        inference_with_cache_time = time.time() - start_time

        self.assertEqual(len(predictions_without_cache), len(predictions_with_cache))
        for p1, p2 in zip(predictions_without_cache, predictions_with_cache):
            self.assertEqual(p1, p2)

        logger.info(
            f"Time of inference without cache: {inference_without_cache_time}, "
            f"with cache (cache is full): {inference_with_cache_time}"
        )

        self.assertGreater(inference_without_cache_time, 2)
        self.assertLess(inference_with_cache_time, 0.5)

        # Ensure that even in the case of failures, the cache allows incremental addition of predictions,
        # enabling the run to complete. To test this, introduce noise that causes the inference engine's
        # `infer` method to return empty results 20% of the time (empty results are not stored in the cache).
        # Verify that after enough runs, all predictions are successfully cached and the final results
        # match those obtained without caching.

        if os.path.exists(unitxt.settings.inference_engine_cache_path):
            shutil.rmtree(unitxt.settings.inference_engine_cache_path)

        inference_model = HFPipelineBasedInferenceEngine(
            model_name=model_name,
            max_new_tokens=32,
            temperature=0,
            top_p=1,
            use_cache=True,
            cache_batch_size=5,
            device="cpu",
        )

        def my_wrapper(original_method):
            random.seed(int(time.time()))

            def wrapped(*args, **kwargs):
                predictions = original_method(*args, **kwargs)
                return [p if random.random() < 0.6 else None for p in predictions]

            return wrapped

        inference_model._infer = my_wrapper(inference_model._infer)
        predictions = [None]
        while predictions.count(None) > 0:
            start_time = time.time()
            predictions = inference_model.infer(dataset)
            inference_time = time.time() - start_time
            logger.info(
                f"Inference time: {inference_time}, predictions contains {predictions.count(None)} Nones"
            )

        self.assertEqual(len(predictions_without_cache), len(predictions_with_cache))
        for p1, p2 in zip(predictions_without_cache, predictions_with_cache):
            self.assertEqual(p1, p2)

    def test_wml_chat_tool_calling(self):
        instance = {
            "source": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "What is 1 + 2?",
                },
            ],
        }

        tool1 = {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "description": "The city, e.g. San Francisco, CA",
                            "type": "string",
                        },
                        "unit": {
                            "enum": ["celsius", "fahrenheit"],
                            "type": "string",
                        },
                    },
                    "required": [
                        "location",
                    ],
                },
            },
        }
        tool2 = {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                        },
                        "b": {
                            "type": "number",
                        },
                    },
                    "required": [
                        "a",
                        "b",
                    ],
                },
            },
        }

        instance["task_data"] = {
            "__tools__": [tool1, tool2],
        }

        dataset = [instance]

        chat = WMLInferenceEngineChat(
            seed=123,
            max_tokens=256,
            temperature=0.0,
            model_name="ibm/granite-3-8b-instruct",
        )

        results = chat.infer(dataset, return_meta_data=False)

        self.assertEqual(results[0], '{"name": "add", "arguments": {"a": 1, "b": 2}}')

    def test_hf_auto_model_and_hf_pipeline_equivalency(self):
        unitxt.settings.allow_unverified_code = True
        for _format in ["formats.chat_api", None]:
            model_name = local_decoder_model  # pragma: allowlist secret
            model_args = {
                "max_new_tokens": 32,
                "temperature": 0,
                "top_p": 1,
                "use_cache": False,
            }

            dataset = load_dataset(
                card="cards.openbook_qa", split="test", format=_format, loader_limit=64
            )  # the number of instances need to large enough to catch differences
            pipeline_inference_model = HFPipelineBasedInferenceEngine(
                model_name=model_name, device="cpu", **model_args
            )
            auto_inference_model = HFAutoModelInferenceEngine(
                model_name=model_name, device_map="cpu", **model_args
            )

            pipeline_inference_model_predictions = pipeline_inference_model.infer(
                dataset
            )
            auto_inference_model_predictions = auto_inference_model.infer(dataset)

            self.assertEqual(
                pipeline_inference_model_predictions, auto_inference_model_predictions
            )
