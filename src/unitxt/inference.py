import abc
import os
from dataclasses import field
from typing import Any, Dict, List, Literal, Optional, Union

from tqdm import tqdm

from .artifact import Artifact
from .operator import PackageRequirementsMixin


class InferenceEngine(abc.ABC, Artifact):
    """Abstract base class for inference."""

    @abc.abstractmethod
    def _infer(self, dataset):
        """Perform inference on the input dataset."""
        pass

    def infer(self, dataset) -> str:
        """Verifies instances of a dataset and performs inference."""
        [self.verify_instance(instance) for instance in dataset]
        return self._infer(dataset)


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


class HFPipelineBasedInferenceEngine(InferenceEngine, PackageRequirementsMixin):
    model_name: str
    max_new_tokens: int
    use_fp16: bool = True
    lazy_load: bool = False

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

    def is_pipeline_initialized(self):
        return hasattr(self, "model") and self.model is not None

    def _infer(self, dataset):
        if not self.is_pipeline_initialized():
            self._prepare_pipeline()

        outputs = []
        for output in self.model([instance["source"] for instance in dataset]):
            if isinstance(output, list):
                output = output[0]
            outputs.append(output["generated_text"])
        return outputs


class MockInferenceEngine(InferenceEngine):
    model_name: str

    def prepare(self):
        return

    def _infer(self, dataset):
        return ["[[10]]" for instance in dataset]


class IbmGenAiInferenceEngineParams(Artifact):
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    typical_p: Optional[float] = None


class IbmGenAiInferenceEngine(InferenceEngine, PackageRequirementsMixin):
    label: str = "ibm_genai"
    model_name: str
    parameters: IbmGenAiInferenceEngineParams = field(
        default_factory=IbmGenAiInferenceEngineParams
    )
    _requirements_list = {
        "genai": "Install ibm-genai package using 'pip install --upgrade ibm-generative-ai"
    }
    data_classification_policy = ["public", "proprietary"]

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

    def _infer(self, dataset):
        from genai.schema import TextGenerationParameters

        genai_params = TextGenerationParameters(
            max_new_tokens=self.parameters.max_new_tokens,
            min_new_tokens=self.parameters.min_new_tokens,
            random_seed=self.parameters.random_seed,
            repetition_penalty=self.parameters.repetition_penalty,
            stop_sequences=self.parameters.stop_sequences,
            temperature=self.parameters.temperature,
            top_p=self.parameters.top_p,
            top_k=self.parameters.top_k,
            typical_p=self.parameters.typical_p,
            decoding_method=self.parameters.decoding_method,
        )

        return [
            response.results[0].generated_text
            for response in self.client.text.generation.create(
                model_id=self.model_name,
                inputs=[instance["source"] for instance in dataset],
                parameters=genai_params,
            )
        ]


class OpenAiInferenceEngineParams(Artifact):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = 20


class OpenAiInferenceEngine(
    InferenceEngine, LogProbInferenceEngine, PackageRequirementsMixin
):
    label: str = "openai"
    model_name: str
    parameters: OpenAiInferenceEngineParams = field(
        default_factory=OpenAiInferenceEngineParams
    )
    _requirements_list = {
        "openai": "Install openai package using 'pip install --upgrade openai"
    }
    data_classification_policy = ["public"]

    def prepare(self):
        from openai import OpenAI

        api_key_env_var_name = "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run OpenAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )

        self.client = OpenAI(api_key=api_key)

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
                frequency_penalty=self.parameters.frequency_penalty,
                presence_penalty=self.parameters.presence_penalty,
                max_tokens=self.parameters.max_tokens,
                seed=self.parameters.seed,
                stop=self.parameters.stop,
                temperature=self.parameters.temperature,
                top_p=self.parameters.top_p,
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
                frequency_penalty=self.parameters.frequency_penalty,
                presence_penalty=self.parameters.presence_penalty,
                max_tokens=self.parameters.max_tokens,
                seed=self.parameters.seed,
                stop=self.parameters.stop,
                temperature=self.parameters.temperature,
                top_p=self.parameters.top_p,
                logprobs=True,
                top_logprobs=self.parameters.top_logprobs,
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

    def initialize_wml_parameters(self) -> Dict[str, Any]:
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

        return {
            param_name.upper(): param_value
            for param_name, param_value in self.to_dict().items()
            if param_value and param_name.upper() in GenTextParamsMetaNames().get()
        }


class WMLInferenceEngine(InferenceEngine, PackageRequirementsMixin):
    """Runs inference using ibm-watsonx-ai.

    Attributes:
        client: By default, it is created by a class instance but can be directly
            provided instead as an instance of 'ibm_watsonx_ai.client.APIClient'.
        credentials: By default, it is created by a class instance which tries to retrieve
            proper environment variables ("WML_URL", "WML_PROJECT_ID", "WML_APIKEY").
            However, either a dictionary with the following keys: "url", "apikey",
            "project_id", or an instance of 'ibm_watsonx_ai.credentials.Credentials'
            can be directly provided instead.
        model_name (str, optional): ID of a model to be used for inference. Mutually
            exclusive with 'deployment_id'.
        deployment_id (str, optional): Deployment ID of a tuned model to be used for
            inference. Mutually exclusive with 'model_name'.
        parameters (WMLInferenceEngineParams): An instance of 'WMLInferenceEngineParams'
            which defines parameters used for inference. All the parameters are optional.

    Examples:
        from .api import load_dataset

        wml_parameters = WMLInferenceEngineParams(top_p=0.5, random_seed=123)
        wml_credentials = {
            "url": "some_url", "project_id": "some_id", "api_key": "some_key"
        }
        model_name = "google/flan-t5-xxl"
        wml_inference = WMLInferenceEngine(
            credentials=wml_credentials,
            parameters=wml_parameters,
            model_name=model_name,
        )

        dataset = load_dataset(
            dataset_query="card=cards.argument_topic,template_card_index=0,loader_limit=5"
        )
        results = wml_inference.infer(dataset["test"])
    """

    client = None
    credentials = None
    model_name: Optional[str] = None
    deployment_id: Optional[str] = None
    parameters: WMLInferenceEngineParams = field(
        default_factory=WMLInferenceEngineParams
    )

    _parameters: Dict[str, Any] = field(default_factory=dict)

    label: str = "wml"
    _requirements_list = {
        "ibm-watsonx-ai": "Install ibm-watsonx-ai package using 'pip install --upgrade ibm-watsonx-ai'. "
        "It is advised to have Python version >=3.10 installed, as at lower version this package "
        "may cause conflicts with other installed packages."
    }

    data_classification_policy = ["proprietary"]

    @staticmethod
    def _read_wml_credentials_from_env() -> Dict[str, str]:
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
        if self.client is None:
            self.client = self._initialize_wml_client()
        self._parameters = self.parameters.initialize_wml_parameters()

    def verify(self):
        assert (
            self.model_name
            or self.deployment_id
            and not (self.model_name and self.deployment_id)
        ), "Either 'model_name' or 'deployment_id' must be specified, but not both at the same time."
        super().verify()

    def _infer(self, dataset):
        from ibm_watsonx_ai.foundation_models import ModelInference

        model = ModelInference(
            model_id=self.model_name,
            deployment_id=self.deployment_id,
            api_client=self.client,
        )

        return [
            model.generate_text(
                prompt=instance["source"],
                params=self._parameters,
            )
            for instance in dataset
        ]
