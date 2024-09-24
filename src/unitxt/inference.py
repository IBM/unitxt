import abc
import os
import re
from typing import Any, Dict, List, Literal, Optional, Union

from tqdm import tqdm

from .artifact import Artifact, fetch_artifact
from .dataclass import InternalField, NonPositionalField
from .deprecation_utils import deprecation
from .image_operators import extract_images
from .logging_utils import get_logger
from .operator import PackageRequirementsMixin
from .settings_utils import get_settings

settings = get_settings()


class InferenceEngine(abc.ABC, Artifact):
    """Abstract base class for inference."""

    @abc.abstractmethod
    def _infer(self, dataset):
        """Perform inference on the input dataset."""
        pass

    @abc.abstractmethod
    def prepare_engine(self):
        """Perform inference on the input dataset."""
        pass

    def prepare(self):
        if not settings.mock_inference_mode:
            self.prepare_engine()

    def infer(self, dataset) -> str:
        """Verifies instances of a dataset and performs inference."""
        [self.verify_instance(instance) for instance in dataset]
        if settings.mock_inference_mode:
            return [instance["source"] for instance in dataset]
        return self._infer(dataset)

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


class LogProbInferenceEngine(abc.ABC, Artifact):
    """Abstract base class for inference with log probs."""

    @abc.abstractmethod
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


class LazyLoadMixin(Artifact):
    lazy_load: bool = NonPositionalField(default=False)

    @abc.abstractmethod
    def _is_loaded(self):
        pass


class HFPipelineBasedInferenceEngine(
    InferenceEngine, PackageRequirementsMixin, LazyLoadMixin
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

    def prepare_engine(self):
        if not self.lazy_load:
            self._prepare_pipeline()

    def _is_loaded(self):
        return hasattr(self, "model") and self.model is not None

    def _infer(self, dataset):
        if not self._is_loaded():
            self._prepare_pipeline()

        outputs = []
        for output in self.model([instance["source"] for instance in dataset]):
            if isinstance(output, list):
                output = output[0]
            outputs.append(output["generated_text"])
        return outputs


class MockInferenceEngine(InferenceEngine):
    model_name: str

    def prepare_engine(self):
        return

    def _infer(self, dataset):
        return ["[[10]]" for instance in dataset]


class MockModeMixin(Artifact):
    mock_mode: bool = False


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


class GenericInferenceEngine(InferenceEngine):
    default: Optional[str] = None

    def prepare_engine(self):
        if "UNITXT_INFERENCE_ENGINE" in os.environ:
            engine_reference = os.environ["UNITXT_INFERENCE_ENGINE"]
        else:
            assert self.default is not None, (
                "GenericInferenceEngine could not be initialized"
                '\nThis is since both the "UNITXT_INFERENCE_ENGINE" environmental variable is not set and no default engine was not inputted.'
                "\nFor example, you can fix it by setting"
                "\nexport UNITXT_INFERENCE_ENGINE=engines.ibm_gen_ai.llama_3_70b_instruct"
                "\nto your ~/.bashrc"
                "\nor passing a similar required engine in the default argument"
            )
            engine_reference = self.default
        self.engine, _ = fetch_artifact(engine_reference)

    def _infer(self, dataset):
        return self.engine._infer(dataset)


class OllamaInferenceEngine(InferenceEngine, PackageRequirementsMixin):
    label: str = "ollama"
    model_name: str
    _requirements_list = {
        "ollama": "Install ollama package using 'pip install --upgrade ollama"
    }
    data_classification_policy = ["public", "proprietary"]

    def prepare_engine(self):
        pass

    def _infer(self, dataset):
        import ollama

        result = [
            ollama.chat(
                model="llama2",
                messages=[
                    {
                        "role": "user",
                        "content": instance["source"],
                    },
                ],
            )
            for instance in dataset
        ]
        return [element["message"]["content"] for element in result]


class IbmGenAiInferenceEngine(
    InferenceEngine, IbmGenAiInferenceEngineParamsMixin, PackageRequirementsMixin
):
    label: str = "ibm_genai"
    model_name: str
    _requirements_list = {
        "genai": "Install ibm-genai package using 'pip install --upgrade ibm-generative-ai"
    }
    data_classification_policy = ["public", "proprietary"]
    parameters: Optional[IbmGenAiInferenceEngineParams] = None

    def prepare_engine(self):
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

    def _infer(self, dataset):
        from genai.schema import TextGenerationParameters

        genai_params = TextGenerationParameters(
            **self.to_dict([IbmGenAiInferenceEngineParamsMixin])
        )

        return [
            response.results[0].generated_text
            for response in self.client.text.generation.create(
                model_id=self.model_name,
                inputs=[instance["source"] for instance in dataset],
                parameters=genai_params,
            )
        ]


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
    logprobs: Optional[bool] = True
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
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
    logprobs: Optional[bool] = True
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    service_tier: Optional[Literal["auto", "default"]] = None


class OpenAiInferenceEngine(
    InferenceEngine,
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

    def prepare_engine(self):
        from openai import OpenAI

        api_key_env_var_name = "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run OpenAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )

        self.client = OpenAI(api_key=api_key)

        self._set_inference_parameters()

    def _get_completion_kwargs(self):
        return {
            k: v
            for k, v in self.to_dict([OpenAiInferenceEngineParamsMixin]).items()
            if v is not None
        }

    def _infer(self, dataset):
        outputs = []
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
                **self._get_completion_kwargs(),
            )
            output = response.choices[0].message.content

            outputs.append(output)

        return outputs

    def _infer_log_probs(self, dataset):
        outputs = []
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
                **self._get_completion_kwargs(),
            )
            top_logprobs_response = response.choices[0].logprobs.content
            output = [
                {
                    "top_tokens": [
                        {"text": obj.token, "logprob": obj.logprob}
                        for obj in generated_token.top_logprobs
                    ]
                }
                for generated_token in top_logprobs_response
            ]
            outputs.append(output)
        return outputs


class TogetherAiInferenceEngineParamsMixin(Artifact):
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = None
    n: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


class TogetherAiInferenceEngine(
    InferenceEngine, TogetherAiInferenceEngineParamsMixin, PackageRequirementsMixin
):
    label: str = "together"
    model_name: str
    _requirements_list = {
        "together": "Install together package using 'pip install --upgrade together"
    }
    data_classification_policy = ["public"]
    parameters: Optional[TogetherAiInferenceEngineParamsMixin] = None

    def prepare_engine(self):
        from together import Together
        from together.types.models import ModelType

        api_key_env_var_name = "TOGETHER_API_KEY"
        api_key = os.environ.get(api_key_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run TogetherAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )
        self.client = Together(api_key=api_key)
        self._set_inference_parameters()

        # Get model type from Together List Models API
        together_models = self.client.models.list()
        together_model_id_to_type = {
            together_model.id: together_model.type for together_model in together_models
        }
        model_type = together_model_id_to_type.get(self.model_name)
        assert model_type is not None, (
            f"Could not find model {self.model_name} " "in Together AI model list"
        )
        assert model_type in [ModelType.CHAT, ModelType.LANGUAGE, ModelType.CODE], (
            f"Together AI model type {model_type} is not supported; "
            "supported types are 'chat', 'language' and 'code'."
        )
        self.model_type = model_type

    def _get_infer_kwargs(self):
        return {
            k: v
            for k, v in self.to_dict([TogetherAiInferenceEngineParamsMixin]).items()
            if v is not None
        }

    def _infer_chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self._get_infer_kwargs(),
        )
        return response.choices[0].message.content

    def _infer_text(self, prompt: str) -> str:
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            **self._get_infer_kwargs(),
        )
        return response.choices[0].text

    def _infer(self, dataset):
        from together.types.models import ModelType

        outputs = []
        if self.model_type == ModelType.CHAT:
            for instance in tqdm(dataset, desc="Inferring with Together AI Chat API"):
                outputs.append(self._infer_chat(instance["source"]))
        else:
            for instance in tqdm(dataset, desc="Inferring with Together AI Text API"):
                outputs.append(self._infer_text(instance["source"]))
        return outputs


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
    InferenceEngine, WMLInferenceEngineParamsMixin, PackageRequirementsMixin
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
        concurrency_limit (int): number of requests that will be sent in parallel, max is 10.

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
    concurrency_limit: int = 10
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

    def prepare_engine(self):
        self._client = self._initialize_wml_client()

        self._set_inference_parameters()

    def _infer(self, dataset):
        from ibm_watsonx_ai.foundation_models import ModelInference

        model = ModelInference(
            model_id=self.model_name,
            deployment_id=self.deployment_id,
            api_client=self._client,
        )

        # the class was previously used with a dataset that is a single instance
        dataset = dataset if isinstance(dataset, list) else [dataset]

        result = [
            model.generate_text(
                prompt=instance["source"],
                params=self.to_dict([WMLInferenceEngineParamsMixin], keep_empty=False),
            )
            for instance in dataset
        ]

        # the class was previously used with a dataset that is a single instance
        return result[0] if not isinstance(dataset, list) else result


class HFLlavaInferenceEngine(InferenceEngine, LazyLoadMixin):
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

    def prepare_engine(self):
        if not self.lazy_load:
            self._prepare_engine()

    def _is_loaded(self):
        return hasattr(self, "model") and self.model is not None

    def _infer(self, dataset):
        if not self._is_loaded():
            self._prepare_engine()

        import torch

        results = []
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
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
            result = self.processor.decode(
                output[0][input_len:], skip_special_tokens=True
            )
            results.append(result)

        return results
