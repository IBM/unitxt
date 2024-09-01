import abc
import json
import os
import re
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Union

from tqdm import tqdm

from .artifact import Artifact
from .dataclass import InternalField, NonPositionalField
from .deprecation_utils import deprecation
from .image_operators import extract_images
from .logging_utils import get_logger
from .operator import PackageRequirementsMixin


class InferenceEngine(Artifact):
    """Base class for inference engines.

    This class provides a framework for processing and inferring data from datasets.
    Subclasses should implement the `_infer_dataset` method to define specific inference behavior.
    """

    def __call__(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes the dataset and applies the inference engine.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries, each representing a data instance.

        Returns:
            List[Dict[str, Any]]: The processed dataset with inferred results.
        """
        processed_dataset = []
        for instance in dataset:
            self.verify_instance(instance)
            if isinstance(instance["task_data"], str):
                instance = {**instance, "task_data": json.loads(instance["task_data"])}
            processed_dataset.append(instance)
        return self._infer_dataset(processed_dataset)

    def infer(self, dataset: List[Dict[str, Any]]) -> List[str]:
        """Infers predictions from the dataset.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries, each representing a data instance.

        Returns:
            List[str]: A list of predictions for each data instance.
        """
        dataset = self(dataset)
        return [instance["prediction"] for instance in dataset]

    @abstractmethod
    def _infer_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Abstract method to perform inference on the dataset.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries, each representing a data instance.

        Returns:
            List[Dict[str, Any]]: The dataset with inferred results.
        """
        pass


class TextGenerationInferenceEngine(InferenceEngine):
    """Abstract base class for text generation inference engines.

    This class is designed for engines that generate text-based outputs from input datasets.
    """

    @abstractmethod
    def generate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Abstract method to perform text generation on the dataset.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries, each representing a data instance.

        Returns:
            List[Dict[str, Any]]: The dataset with generated text results.
        """
        pass

    def _infer_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.generate(dataset)

    @deprecation(version="2.0.0")
    def _set_inference_parameters(self):
        """Sets inference parameters of an instance based on 'parameters' attribute (if given)."""
        if hasattr(self, "parameters") and self.parameters is not None:
            get_logger().warning(
                f"The 'parameters' attribute of '{self.get_pretty_print_name()}' "
                f"is deprecated. Please pass inference parameters directly to the "
                f"inference engine instance instead."
            )

            for param, param_dict_val in self.parameters.to_dict(
                [self.parameters]
            ).items():
                param_inst_val = getattr(self, param)
                if param_inst_val is None:
                    setattr(self, param, param_dict_val)


class ScoringInferenceEngine(InferenceEngine):
    """Abstract class for inference engines that assign scores to texts.

    This class is designed for engines that produce a score for each input instance in the dataset.
    """

    def _infer_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verifies instances of a dataset and performs inference."""
        return self.score(dataset)

    @abstractmethod
    def score(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Abstract method to assign scores to the dataset.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries, each representing a data instance.

        Returns:
            List[Dict[str, Any]]: The dataset with scores assigned to each instance.
        """


class OptionSelectingInferenceEngine(InferenceEngine):
    """Abstract class for inference engines that select options based on inference results.

    This class is designed for engines that select the best option from a set of options for each input instance.
    """

    def _infer_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.select(dataset)

    @abstractmethod
    def select(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Abstract method to select options from the dataset.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries, each representing a data instance.

        Returns:
            List[Dict[str, Any]]: The dataset with selected options for each instance.
        """
        pass


class HFLogProbScoringEngine(ScoringInferenceEngine, PackageRequirementsMixin):
    """HuggingFace based class for inference engines that calculate log probabilities.

    This class uses models from the HuggingFace Transformers library to calculate log probabilities for text inputs.
    """

    model_name: str
    batch_size: int

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers"
    }

    def prepare(self):
        super().prepare()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.device
        )
        # Set pad_token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_log_probs(self, texts):
        # Check available device
        import torch
        from tqdm import tqdm

        log_probs = []

        # Process texts in batches
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i : i + self.batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            # Compute log probabilities
            with torch.no_grad():
                predictions = self.model(**inputs)
                logits = predictions.logits

                for j in range(len(batch)):
                    input_ids = inputs.input_ids[j]
                    text_logits = logits[j, :-1, :]  # exclude last token
                    text_log_probs = torch.log_softmax(text_logits, dim=-1)

                    # Gather log probs for each token
                    token_log_probs = text_log_probs[
                        torch.arange(text_logits.shape[0]), input_ids[1:]
                    ]

                    # Sum log probs to get sequence log prob
                    sequence_log_prob = token_log_probs.sum().item()
                    log_probs.append(sequence_log_prob)

        return log_probs

    def score(self, dataset):
        """Add to each instance in the data a "prediction" score field."""
        texts = []
        for instance in dataset:
            text = instance["source"]
            texts.append(text)

        scores = self.get_log_probs(texts)

        for instance, score in zip(dataset, scores):
            instance["prediction"] = score

        return dataset


class SelectingByScoreInferenceEngine(OptionSelectingInferenceEngine):
    scorer_engine: ScoringInferenceEngine

    def select(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dataset_to_score = []

        for instance in dataset:
            for option in instance["task_data"]["options"]:
                dataset_to_score.append({"source": instance["source"] + option})

        scores = self.scorer_engine.score(dataset_to_score)

        scores_iterator = iter(scores)

        for instance in dataset:
            options_scores = Counter()
            for option in instance["task_data"]["options"]:
                score = next(scores_iterator)["prediction"]
                options_scores[option] = score
            instance["prediction"] = options_scores.most_common(1)[0][0]

        return dataset


class LogProbInferenceEngine(Artifact):
    """Abstract base class for inference with log probs."""

    @abstractmethod
    def _infer_log_probs(self, dataset):
        """Perform inference on the input dataset that returns log probs."""
        pass

    def infer_log_probs(self, dataset) -> List[Dict]:
        """Verifies instances of a dataset and performs inference that returns log probabilities of top tokens.

        For each instance , returns a list of top tokens per position.
        [ "top_tokens": [ { "text": ..., "logprob": ...} , ... ]

        """
        [self.verify_instance(instance) for instance in dataset]
        return self._infer_log_probs(dataset)


class HFLogProbInferenceEngine(LogProbInferenceEngine):
    """Abstract base class for inference with log probs."""

    model_name: str
    batch_size: int

    def prepare(self):
        super().prepare()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device=self.device
        )
        # Set pad_token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_log_probs(self, texts):
        # Check available device
        import torch
        from tqdm import tqdm

        log_probs = []

        # Process texts in batches
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i : i + self.batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            # Compute log probabilities
            with torch.no_grad():
                predictions = self.model(**inputs)
                logits = predictions.logits

                for j in range(len(batch)):
                    input_ids = inputs.input_ids[j]
                    text_logits = logits[j, :-1, :]  # exclude last token
                    text_log_probs = torch.log_softmax(text_logits, dim=-1)

                    # Gather log probs for each token
                    token_log_probs = text_log_probs[
                        torch.arange(text_logits.shape[0]), input_ids[1:]
                    ]

                    # Sum log probs to get sequence log prob
                    sequence_log_prob = token_log_probs.sum().item()
                    log_probs.append(sequence_log_prob)

        return log_probs

    def _infer_log_probs(self, dataset):
        """Perform inference on the input dataset that returns log probs."""
        pass


class LazyLoadMixin(Artifact):
    lazy_load: bool = NonPositionalField(default=False)

    @abc.abstractmethod
    def _is_loaded(self):
        pass


class HFPipelineBasedInferenceEngine(
    TextGenerationInferenceEngine, PackageRequirementsMixin, LazyLoadMixin
):
    model_name: str
    max_new_tokens: int
    use_fp16: bool = True

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers"
    }

    def _prepare_pipeline(self):
        import torch
        from transformers import AutoConfig, pipeline

        model_args: Dict[str, Any] = (
            {"torch_dtype": torch.float16} if self.use_fp16 else {}
        )
        model_args.update({"max_new_tokens": self.max_new_tokens})

        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else 0
            if torch.cuda.is_available()
            else "cpu"
        )
        # We do this, because in some cases, using device:auto will offload some weights to the cpu
        # (even though the model might *just* fit to a single gpu), even if there is a gpu available, and this will
        # cause an error because the data is always on the gpu
        if torch.cuda.device_count() > 1:
            assert device == torch.device(0)
            model_args.update({"device_map": "auto"})
        else:
            model_args.update({"device": device})

        task = (
            "text2text-generation"
            if AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            ).is_encoder_decoder
            else "text-generation"
        )

        if task == "text-generation":
            model_args.update({"return_full_text": False})

        self.model = pipeline(
            model=self.model_name, trust_remote_code=True, **model_args
        )

    def prepare(self):
        if not self.lazy_load:
            self._prepare_pipeline()

    def _is_loaded(self):
        return hasattr(self, "model") and self.model is not None

    def generate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._is_loaded():
            self._prepare_pipeline()

        predictions = self.model([instance["source"] for instance in dataset])

        for prediction, instance in zip(predictions, dataset):
            if isinstance(prediction, list):
                prediction = prediction[0]
            instance["prediction"] = prediction["generated_text"]

        return dataset


class MockInferenceEngine(InferenceEngine):
    model_name: str

    def prepare(self):
        return

    def _infer(self, dataset):
        return ["[[10]]" for instance in dataset]


class IbmGenAiInferenceEngineParamsMixin(Artifact):
    beam_width: Optional[int] = None
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    include_stop_sequence: Optional[bool] = None
    length_penalty: Any = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_options: Any = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    time_limit: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    truncate_input_tokens: Optional[int] = None
    typical_p: Optional[float] = None


@deprecation(version="2.0.0", alternative=IbmGenAiInferenceEngineParamsMixin)
class IbmGenAiInferenceEngineParams(Artifact):
    beam_width: Optional[int] = None
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    include_stop_sequence: Optional[bool] = None
    length_penalty: Any = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_options: Any = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    time_limit: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    truncate_input_tokens: Optional[int] = None
    typical_p: Optional[float] = None


class IbmGenAiInferenceEngine(
    TextGenerationInferenceEngine,
    IbmGenAiInferenceEngineParamsMixin,
    PackageRequirementsMixin,
):
    label: str = "ibm_genai"
    model_name: str
    _requirements_list = {
        "genai": "Install ibm-genai package using 'pip install --upgrade ibm-generative-ai"
    }
    data_classification_policy = ["public", "proprietary"]
    parameters: Optional[IbmGenAiInferenceEngineParams] = None

    def prepare(self):
        from genai import Client, Credentials

        api_key_env_var_name = "GENAI_KEY"
        api_key = os.environ.get(api_key_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run IbmGenAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )
        credentials = Credentials(api_key=api_key)
        self.client = Client(credentials=credentials)

        self._set_inference_parameters()

    def generate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from genai.schema import TextGenerationParameters

        genai_params = TextGenerationParameters(
            **self.to_dict([IbmGenAiInferenceEngineParamsMixin])
        )

        predictions = [
            response.results[0].generated_text
            for response in self.client.text.generation.create(
                model_id=self.model_name,
                inputs=[instance["source"] for instance in dataset],
                parameters=genai_params,
            )
        ]

        for instance, prediction in zip(dataset, predictions):
            instance["prediction"] = prediction

        return dataset


class OpenAiInferenceEngineParamsMixin(Artifact):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = 20
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    n: Optional[int] = None
    parallel_tool_calls: bool = None
    service_tier: Optional[Literal["auto", "default"]] = None


@deprecation(version="2.0.0", alternative=OpenAiInferenceEngineParamsMixin)
class OpenAiInferenceEngineParams(Artifact):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = 20
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    n: Optional[int] = None
    parallel_tool_calls: bool = None
    service_tier: Optional[Literal["auto", "default"]] = None


class OpenAiInferenceEngine(
    TextGenerationInferenceEngine,
    LogProbInferenceEngine,
    OpenAiInferenceEngineParamsMixin,
    PackageRequirementsMixin,
):
    label: str = "openai"
    model_name: str
    _requirements_list = {
        "openai": "Install openai package using 'pip install --upgrade openai"
    }
    data_classification_policy = ["public"]
    parameters: Optional[OpenAiInferenceEngineParams] = None

    def prepare(self):
        from openai import OpenAI

        api_key_env_var_name = "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run OpenAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )

        self.client = OpenAI(api_key=api_key)

        self._set_inference_parameters()

    def generate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        predictions = []
        for instance in tqdm(dataset, desc="Inferring with openAI API"):
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": instance["source"],
                    }
                ],
                model=self.model_name,
                **self.to_dict([OpenAiInferenceEngineParamsMixin]),
            )
            prediction = response.choices[0].message.content

            predictions.append(prediction)

        for instance, prediction in zip(dataset, predictions):
            instance["prediction"] = prediction

        return dataset

    def _infer_log_probs(self, dataset):
        predictions = []
        for instance in tqdm(dataset, desc="Inferring with openAI API"):
            response = self.client.chat.completions.create(
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": self.system_prompt,
                    # },
                    {
                        "role": "user",
                        "content": instance["source"],
                    }
                ],
                model=self.model_name,
                **self.to_dict([OpenAiInferenceEngineParamsMixin]),
            )
            top_logprobs_response = response.choices[0].logprobs.content
            prediction = [
                {
                    "top_tokens": [
                        {"text": obj.token, "logprob": obj.logprob}
                        for obj in generated_token.top_logprobs
                    ]
                }
                for generated_token in top_logprobs_response
            ]
            predictions.append(prediction)
        return predictions


class WMLInferenceEngineParamsMixin(Artifact):
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    length_penalty: Optional[Dict[str, Union[int, float]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    min_new_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    time_limit: Optional[int] = None
    truncate_input_tokens: Optional[int] = None
    prompt_variables: Optional[Dict[str, Any]] = None
    return_options: Optional[Dict[str, bool]] = None


@deprecation(version="2.0.0", alternative=WMLInferenceEngineParamsMixin)
class WMLInferenceEngineParams(Artifact):
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    length_penalty: Optional[Dict[str, Union[int, float]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    min_new_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    time_limit: Optional[int] = None
    truncate_input_tokens: Optional[int] = None
    prompt_variables: Optional[Dict[str, Any]] = None
    return_options: Optional[Dict[str, bool]] = None


class WMLInferenceEngine(
    TextGenerationInferenceEngine,
    WMLInferenceEngineParamsMixin,
    PackageRequirementsMixin,
):
    """Runs inference using ibm-watsonx-ai.

    Attributes:
        credentials (Dict[str, str], optional): By default, it is created by a class
            instance which tries to retrieve proper environment variables
            ("WML_URL", "WML_PROJECT_ID", "WML_APIKEY"). However, a dictionary with
            the following keys: "url", "apikey", "project_id" can be directly provided
            instead.
        model_name (str, optional): ID of a model to be used for inference. Mutually
            exclusive with 'deployment_id'.
        deployment_id (str, optional): Deployment ID of a tuned model to be used for
            inference. Mutually exclusive with 'model_name'.
        parameters (WMLInferenceEngineParams, optional): Instance of WMLInferenceEngineParams
            which defines inference parameters and their values. Deprecated attribute, please
            pass respective parameters directly to the WMLInferenceEngine class instead.

    Examples:
        from .api import load_dataset

        wml_credentials = {
            "url": "some_url", "project_id": "some_id", "api_key": "some_key"
        }
        model_name = "google/flan-t5-xxl"
        wml_inference = WMLInferenceEngine(
            credentials=wml_credentials,
            model_name=model_name,
            data_classification_policy=["public"],
            top_p=0.5,
            random_seed=123,
        )

        dataset = load_dataset(
            dataset_query="card=cards.argument_topic,template_card_index=0,loader_limit=5"
        )
        results = wml_inference.infer(dataset["test"])
    """

    credentials: Optional[Dict[Literal["url", "apikey", "project_id"], str]] = None
    model_name: Optional[str] = None
    deployment_id: Optional[str] = None
    label: str = "wml"
    _requirements_list = {
        "ibm_watsonx_ai": "Install ibm-watsonx-ai package using 'pip install --upgrade ibm-watsonx-ai'. "
        "It is advised to have Python version >=3.10 installed, as at lower version this package "
        "may cause conflicts with other installed packages."
    }
    data_classification_policy = ["public", "proprietary"]
    parameters: Optional[WMLInferenceEngineParams] = None

    _client: Any = InternalField(default=None, name="WML client")

    def verify(self):
        super().verify()

        if self.credentials is not None:
            for key in self.credentials:
                if key not in ["url", "apikey", "project_id"]:
                    raise ValueError(
                        f'Illegal credential key: {key}, use only ["url", "apikey", "project_id"]'
                    )

        assert (
            self.model_name
            or self.deployment_id
            and not (self.model_name and self.deployment_id)
        ), "Either 'model_name' or 'deployment_id' must be specified, but not both at the same time."

    def process_data_before_dump(self, data):
        if "credentials" in data:
            for key, value in data["credentials"].items():
                if key != "url":
                    data["credentials"][key] = "<hidden>"
                else:
                    data["credentials"][key] = value
        return data

    @staticmethod
    def _read_wml_credentials_from_env() -> (
        Dict[Literal["url", "apikey", "project_id"], str]
    ):
        credentials = {}
        for env_var_name in ["WML_URL", "WML_PROJECT_ID", "WML_APIKEY"]:
            env_var = os.environ.get(env_var_name)
            assert env_var, (
                f"Error while trying to run 'WMLInferenceEngine'. "
                f"Please set the env variable: '{env_var_name}', or "
                f"directly provide an instance of ibm-watsonx-ai 'Credentials' "
                f"to the engine."
            )

            name = env_var_name.lower().replace("wml_", "")
            credentials[name] = env_var

        return credentials

    def _initialize_wml_client(self):
        from ibm_watsonx_ai.client import APIClient

        if self.credentials is None:
            self.credentials = self._read_wml_credentials_from_env()

        client = APIClient(credentials=self.credentials)
        client.set.default_project(self.credentials["project_id"])
        return client

    def prepare(self):
        self._client = self._initialize_wml_client()

        self._set_inference_parameters()

    def generate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from ibm_watsonx_ai.foundation_models import ModelInference

        model = ModelInference(
            model_id=self.model_name,
            deployment_id=self.deployment_id,
            api_client=self._client,
        )

        predictions = model.generate_text(
            prompt=dataset["source"],
            params=self.to_dict([WMLInferenceEngineParamsMixin], keep_empty=False),
        )

        for instance, prediction in zip(dataset, predictions):
            instance["prediction"] = prediction

        return dataset


class HFLlavaInferenceEngine(TextGenerationInferenceEngine, LazyLoadMixin):
    model_name: str
    max_new_tokens: int
    lazy_load = True

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers",
        "torch": "Install torch, go on PyTorch website for mode details.",
        "accelerate": "pip install accelerate",
    }

    def _prepare_engine(self):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else 0
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def prepare(self):
        if not self.lazy_load:
            self._prepare_engine()

    def _is_loaded(self):
        return hasattr(self, "model") and self.model is not None

    def generate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._is_loaded():
            self._prepare_engine()

        import torch

        predictions = []
        for instance in dataset:
            text = instance["source"]
            images = extract_images(text, instance)
            # Regular expression to match all <img src="..."> tags
            regex = r'<img\s+src=["\'](.*?)["\']\s*/?>'
            model_input = re.sub(regex, "<image>", text)
            if len(images) == 1:
                images = images[0]
            inputs = self.processor(
                images=images, text=model_input, return_tensors="pt"
            ).to(self.device, torch.float16)
            input_len = len(inputs["input_ids"][0])
            prediction = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
            prediction = self.processor.decode(
                prediction[0][input_len:], skip_special_tokens=True
            )
            predictions.append(prediction)

        for instance, prediction in zip(dataset, predictions):
            instance["prediction"] = prediction

        return dataset
