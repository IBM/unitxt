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
    def _infer(self, dataset, return_meta_data=False):
        """Perform inference on the input dataset."""
        pass

    @abc.abstractmethod
    def prepare_engine(self):
        """Perform inference on the input dataset."""
        pass

    def prepare(self):
        if not settings.mock_inference_mode:
            self.prepare_engine()

    def infer(self, dataset, return_meta_data=False) -> str:
        """Verifies instances of a dataset and performs inference."""
        assert return_meta_data is False or hasattr(self, "get_return_object"), (
            f"Inference engin {self.__class__.__name__} does not support return_meta_data,"
            f"Please set return_meta_data=False"
        )

        [self.verify_instance(instance) for instance in dataset]
        if settings.mock_inference_mode:
            return [instance["source"] for instance in dataset]
        return self._infer(dataset, return_meta_data)

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
    def _infer_log_probs(self, dataset, return_meta_data=False):
        """Perform inference on the input dataset that returns log probs."""
        pass

    def infer_log_probs(self, dataset, return_meta_data=False) -> List[Dict]:
        """Verifies instances of a dataset and performs inference that returns log probabilities of top tokens.

        For each instance , returns a list of top tokens per position.
        [ "top_tokens": [ { "text": ..., "logprob": ...} , ... ]

        """
        assert return_meta_data is False or hasattr(self, "get_return_object"), (
            f"Inference engin {self.__class__.__name__} does not support return_meta_data,"
            f"Please set return_meta_data=False"
        )

        [self.verify_instance(instance) for instance in dataset]
        return self._infer_log_probs(dataset, return_meta_data)


class LazyLoadMixin(Artifact):
    lazy_load: bool = NonPositionalField(default=False)

    @abc.abstractmethod
    def _is_loaded(self):
        pass


class InferenceEngineReturn:
    """Contains the prediction results and metadata for the inference.

    Args:
    prediction (Union[str, List[Dict[str, Any]]]): If this is the result of an _infer call, the string predicted by the model.
    If this is the results of an _infer_log_probs call, a list of dictionaries. The i'th dictionary represents
    the i'th token in the response. The entry "top_tokens" in the dictionary holds a sorted list of the top tokens
    for this position and their probabilities.
    For example: [ {.. "top_tokens": [ {"text": "a", 'logprob': -1.42},  {"text": "b", 'logprob': -2.3} ....]},
                   {.. "top_tokens": [ {"text": "c", 'logprob': -0.8},  {"text": "b", 'logprob': -3.3} ....]}
                ]
    means that for the first token in the output, the most probable token is "a" with logprob -1.42, followed by "b"
    with 'logprob': -2.3}. For the second token in the response, the most probable token is "c" etc.

    input_tokens (int) : number of input tokens to the model.
    output_tokens (int) : number of output tokens to the model.
    model_name (str): the model_name as kept in the InferenceEngine.
    inference_type (str): The label stating the type of the InferenceEngine.
    """

    prediction: Union[str, List[Dict[str, Any]]]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    model_name: Optional[str]
    inference_type: Optional[str]

    def __init__(
        self,
        prediction: Union[str, Dict[str, Any]],
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        inference_type: Optional[str] = None,
    ):
        self.prediction = prediction
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model_name = model_name
        self.inference_type = inference_type


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

    def _infer(self, dataset, return_meta_data=False):
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

    def _infer(self, dataset, return_meta_data=False):
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

    def _infer(self, dataset, return_meta_data=False):
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

    def _infer(self, dataset, return_meta_data=False):
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
    InferenceEngine,
    IbmGenAiInferenceEngineParamsMixin,
    PackageRequirementsMixin,
    LogProbInferenceEngine,
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

    def _infer(self, dataset, return_meta_data=False):
        from genai.schema import TextGenerationParameters

        genai_params = TextGenerationParameters(
            **self.to_dict([IbmGenAiInferenceEngineParamsMixin])
        )

        results = []
        responses = self.client.text.generation.create(
            model_id=self.model_name,
            inputs=[instance["source"] for instance in dataset],
            parameters=genai_params,
        )
        for response in responses:
            generated_text = response.results[0].generated_text
            result = self.get_return_object(
                generated_text, response.results[0], return_meta_data
            )
            results.append(result)
        return results

    def _infer_log_probs(self, dataset, return_meta_data=False):
        from genai.schema import TextGenerationParameters

        logprobs_return_options = {
            "generated_tokens": True,
            "input_text": False,
            "input_tokens": False,
            "token_logprobs": True,
            "token_ranks": True,
            "top_n_tokens": 5,
        }
        genai_params = self.to_dict(
            [IbmGenAiInferenceEngineParamsMixin], keep_empty=False
        )
        genai_params = {**genai_params, "return_options": logprobs_return_options}
        genai_params = TextGenerationParameters(**genai_params)
        predictions = self.client.text.generation.create(
            model_id=self.model_name,
            inputs=[instance["source"] for instance in dataset],
            parameters=genai_params,
        )

        predict_results = []
        for prediction in predictions:
            result = prediction.results[0]
            assert isinstance(
                result.generated_tokens, list
            ), "result.generated_tokens should be a list"

            predict_result = []
            for base_token in result.generated_tokens:
                res = {**base_token.__dict__, **base_token.model_extra}
                res["top_tokens"] = [
                    {"logprob": top_token.logprob, "text": top_token.text}
                    for top_token in res["top_tokens"]
                ]
                predict_result.append(res)
            final_results = self.get_return_object(
                predict_result, result, return_meta_data
            )
            predict_results.append(final_results)
        return predict_results

    def get_return_object(self, predict_result, result, return_meta_data):
        if return_meta_data:
            return InferenceEngineReturn(
                prediction=predict_result,
                input_tokens=result.input_token_count,
                output_tokens=result.generated_token_count,
                model_name=self.model_name,
                inference_type=self.label,
            )
        return predict_result


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

    @classmethod
    def get_api_param(cls, inference_engine: str, api_param_env_var_name: str):
        api_key = os.environ.get(api_param_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run {inference_engine}."
            f" Please set the environment param '{api_param_env_var_name}'."
        )
        return api_key

    def create_client(self):
        from openai import OpenAI

        api_key = self.get_api_param(
            inference_engine="OpenAiInferenceEngine",
            api_param_env_var_name="OPENAI_API_KEY",
        )
        return OpenAI(api_key=api_key)

    def prepare_engine(self):
        self.client = self.create_client()
        self._set_inference_parameters()

    def _get_completion_kwargs(self):
        return {
            k: v
            for k, v in self.to_dict([OpenAiInferenceEngineParamsMixin]).items()
            if v is not None
        }

    def _infer(self, dataset, return_meta_data=False):
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
            prediction = response.choices[0].message.content
            output = self.get_return_object(prediction, response, return_meta_data)

            outputs.append(output)

        return outputs

    def _infer_log_probs(self, dataset, return_meta_data=False):
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
            pred_output = [
                {
                    "top_tokens": [
                        {"text": obj.token, "logprob": obj.logprob}
                        for obj in generated_token.top_logprobs
                    ]
                }
                for generated_token in top_logprobs_response
            ]
            output = self.get_return_object(pred_output, response, return_meta_data)
            outputs.append(output)
        return outputs

    def get_return_object(self, predict_result, response, return_meta_data):
        if return_meta_data:
            return InferenceEngineReturn(
                prediction=predict_result,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model_name=self.model_name,
                inference_type=self.label,
            )
        return predict_result


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

    def _infer(self, dataset, return_meta_data=False):
        from together.types.models import ModelType

        outputs = []
        if self.model_type == ModelType.CHAT:
            for instance in tqdm(dataset, desc="Inferring with Together AI Chat API"):
                outputs.append(self._infer_chat(instance["source"]))
        else:
            for instance in tqdm(dataset, desc="Inferring with Together AI Text API"):
                outputs.append(self._infer_text(instance["source"]))
        return outputs


class VLLMRemoteInferenceEngine(OpenAiInferenceEngine):
    def create_client(self):
        from openai import OpenAI

        api_key = self.get_api_param(
            inference_engine="VLLMRemoteInferenceEngine",
            api_param_env_var_name="VLLM_API_KEY",
        )
        api_url = self.get_api_param(
            inference_engine="VLLMRemoteInferenceEngine",
            api_param_env_var_name="VLLM_API_URL",
        )
        return OpenAI(api_key=api_key, base_url=api_url)


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
    InferenceEngine,
    WMLInferenceEngineParamsMixin,
    PackageRequirementsMixin,
    LogProbInferenceEngine,
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
                if key not in ["url", "apikey", "project_id", "space_id"]:
                    raise ValueError(
                        f'Illegal credential key: {key}, use only ["url", "apikey", "project_id", "space_id"]'
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
        Dict[Literal["url", "apikey", "project_id", "space_id"], str]
    ):
        credentials = {}
        project_or_deployment_var_name = (
            "WML_SPACE_ID" if "WML_SPACE_ID" in os.environ else "WML_PROJECT_ID"
        )

        for env_var_name in ["WML_URL", project_or_deployment_var_name, "WML_APIKEY"]:
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
        if "space_id" in self.credentials:
            client.set.default_space(self.credentials["space_id"])
        else:
            client.set.default_project(self.credentials["project_id"])
        return client

    def prepare_engine(self):
        self._client = self._initialize_wml_client()

        self._set_inference_parameters()

    def _load_model_and_params(self):
        from ibm_watsonx_ai.foundation_models import ModelInference

        model = ModelInference(
            model_id=self.model_name,
            deployment_id=self.deployment_id,
            api_client=self._client,
        )
        params = self.to_dict([WMLInferenceEngineParamsMixin], keep_empty=False)

        return model, params

    def _infer(self, dataset, return_meta_data=False):
        model, params = self._load_model_and_params()

        result = []
        for instance in dataset:
            instance_result = model.generate(
                prompt=instance["source"],
                params=self.to_dict([WMLInferenceEngineParamsMixin], keep_empty=False),
            )
            prediction = instance_result["results"][0]["generated_text"]
            instance_final_results = self.get_return_object(
                prediction, instance_result, return_meta_data
            )
            result.append(instance_final_results)

        return result

    def _infer_log_probs(self, dataset, return_meta_data=False):
        model, params = self._load_model_and_params()

        user_return_options = params.pop("return_options", {})
        # currently this is the only configuration that returns generated logprobs and behaves as expected
        logprobs_return_options = {
            "input_tokens": True,
            "generated_tokens": True,
            "token_logprobs": True,
            "top_n_tokens": user_return_options.get("top_n_tokens", 5),
        }
        for key, value in logprobs_return_options.items():
            if key in user_return_options and user_return_options[key] != value:
                raise ValueError(
                    f"'{key}={user_return_options[key]}' is not supported for the 'infer_log_probs' "
                    f"method of {self.__class__.__name__}. For obtaining the logprobs of generated tokens "
                    f"please use '{key}={value}'."
                )

        params = {
            **params,
            "return_options": logprobs_return_options,
        }

        results = model.generate(
            prompt=dataset["source"],
            params=params,
        )
        final_results = []
        for result in results:
            generated_tokens = result["results"][0]["generated_tokens"]
            final_results.append(
                self.get_return_object(generated_tokens, result, return_meta_data)
            )
        return final_results

    def get_return_object(self, predict_result, result, return_meta_data):
        if return_meta_data:
            return InferenceEngineReturn(
                prediction=predict_result,
                input_tokens=result["results"][0]["input_token_count"],
                output_tokens=result["results"][0]["generated_token_count"],
                model_name=self.model_name,
                inference_type=self.label,
            )
        return predict_result


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

    def _infer(self, dataset, return_meta_data=False):
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
