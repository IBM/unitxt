import abc
import asyncio
import dataclasses
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Union

from datasets import DatasetDict
from tqdm import tqdm, trange
from tqdm.asyncio import tqdm_asyncio

from .artifact import Artifact
from .dataclass import InternalField, NonPositionalField
from .deprecation_utils import deprecation
from .error_utils import UnitxtError
from .image_operators import data_url_to_image, extract_images
from .logging_utils import get_logger
from .operator import PackageRequirementsMixin
from .operators import ArtifactFetcherMixin
from .settings_utils import get_constants, get_settings

constants = get_constants()
settings = get_settings()
logger = get_logger()


class StandardAPIParamsMixin(Artifact):
    model: str
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    service_tier: Optional[Literal["auto", "default"]] = None


def get_model_and_label_id(model_name, label):
    model_id = model_name.split("/")[-1].replace("-", "_").replace(".", ",").lower()
    return f"{model_id}_{label}"


@dataclasses.dataclass
class TextGenerationInferenceOutput:
    """Contains the prediction results and metadata for the inference.

    Args:
    prediction (Union[str, List[Dict[str, Any]]]): If this is the result of an _infer call, the string predicted by the model.
    If this is the results of an _infer_log_probs call, a list of dictionaries. The i'th dictionary represents
    the i'th token in the response. The entry "top_tokens" in the dictionary holds a sorted list of the top tokens
    for this position and their probabilities.
    For example: [ {.. "top_tokens": [ {"text": "a", 'logprob': },  {"text": "b", 'logprob': } ....]},
                   {.. "top_tokens": [ {"text": "c", 'logprob': },  {"text": "d", 'logprob': } ....]}
                ]

    input_tokens (int) : number of input tokens to the model.
    output_tokens (int) : number of output tokens to the model.
    model_name (str): the model_name as kept in the InferenceEngine.
    inference_type (str): The label stating the type of the InferenceEngine.
    """

    prediction: Union[str, List[Dict[str, Any]]]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model_name: Optional[str] = None
    inference_type: Optional[str] = None


class InferenceEngine(Artifact):
    """Abstract base class for inference."""

    @abc.abstractmethod
    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        """Perform inference on the input dataset.

        If return_meta_data - returns a list of TextGenerationInferenceOutput, else returns a list of the string.
        return_meta_data is only supported for some InferenceEngines.
        predictions.
        """
        pass

    @abc.abstractmethod
    def prepare_engine(self):
        """Perform inference on the input dataset."""
        pass

    def prepare(self):
        if not settings.mock_inference_mode:
            super().prepare()  # no need to prepare a mock
            self.prepare_engine()

    def infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        """Verifies instances of a dataset and perform inference on the input dataset.

        If return_meta_data - returns a list of TextGenerationInferenceOutput, else returns a list of the string
        predictions.
        """
        if return_meta_data and not hasattr(self, "get_return_object"):
            raise NotImplementedError(
                f"Inference engine {self.__class__.__name__} does not support return_meta_data as it "
                f"does not contain a 'get_return_object' method. Please set return_meta_data=False."
            )

        [self.verify_instance(instance) for instance in dataset]
        if settings.mock_inference_mode:
            return self._mock_infer(dataset)
        return self._infer(dataset, return_meta_data)

    def _mock_infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return [str(instance["source"]) for instance in dataset]

    def get_engine_id(self):
        raise NotImplementedError()

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

    def verify_not_chat_api(self, dataset):
        if isinstance(dataset[0]["source"], list):
            raise NotImplementedError(
                f"Inference engine {self.__class__.__name__} does not support chat api format."
            )

    def to_messages(self, instance):
        if isinstance(instance["source"], list):
            return instance["source"]
        return [
            {
                "role": "user",
                "content": instance["source"],
            }
        ]


class LogProbInferenceEngine(abc.ABC, Artifact):
    """Abstract base class for inference with log probs."""

    @abc.abstractmethod
    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        """Perform inference on the input dataset  that returns log probs.

        If return_meta_data - returns a list of TextGenerationInferenceOutput, else returns a list of the logprob dicts.
        return_meta_data is only supported for some InferenceEngines.
        predictions.
        """
        pass

    def infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        """Verifies instances of a dataset and performs inference that returns log probabilities of top tokens.

        For each instance , generates a list of top tokens per position.
        [ "top_tokens": [ { "text": ..., "logprob": ...} , ... ]
        If return_meta_data - returns a list of TextGenerationInferenceOutput, else returns the list of the logprob dicts.
        return_meta_data is only supported for some InferenceEngines.
        """
        if return_meta_data and not hasattr(self, "get_return_object"):
            raise NotImplementedError(
                f"Inference engine {self.__class__.__name__} does not support return_meta_data as it "
                f"does not contain a 'get_return_object' method. Please set return_meta_data=False."
            )

        [self.verify_instance(instance) for instance in dataset]
        return self._infer_log_probs(dataset, return_meta_data)


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
    batch_size: int = 1
    top_k: Optional[int] = None

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers"
    }

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, "hf_pipeline")

    def _get_task(self):
        from transformers import AutoConfig

        return (
            "text2text-generation"
            if AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            ).is_encoder_decoder
            else "text-generation"
        )

    def _prepare_pipeline(self):
        import torch
        from transformers import pipeline

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

        task = self._get_task()

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

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if self._get_task() == "text2text-generation":
            self.verify_not_chat_api(dataset)

        if not self._is_loaded():
            self._prepare_pipeline()

        outputs = []
        for output in self.model(
            [instance["source"] for instance in dataset],
            batch_size=self.batch_size,
            top_k=self.top_k,
        ):
            if isinstance(output, list):
                output = output[0]
            outputs.append(output["generated_text"])
        return outputs


class MockInferenceEngine(InferenceEngine):
    model_name: str
    default_inference_value: str = "[[10]]"

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, "mock")

    def prepare_engine(self):
        return

    def _mock_infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return [self.default_inference_value for _ in dataset]

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return self._mock_infer(dataset)


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


class GenericInferenceEngine(InferenceEngine, ArtifactFetcherMixin):
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
        self.engine = self.get_artifact(engine_reference)

    def get_engine_id(self):
        return "generic_inference_engine"

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return self.engine._infer(dataset)


class OllamaInferenceEngine(
    InferenceEngine, StandardAPIParamsMixin, PackageRequirementsMixin
):
    label: str = "ollama"
    _requirements_list = {
        "ollama": "Install ollama package using 'pip install --upgrade ollama"
    }
    data_classification_policy = ["public", "proprietary"]

    def get_engine_id(self):
        return get_model_and_label_id(self.model, self.label)

    def prepare_engine(self):
        pass

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        import ollama

        args = self.to_dict([StandardAPIParamsMixin])

        results = []

        for instance in dataset:
            messages = self.to_messages(instance)
            response = ollama.chat(
                model=self.model,
                messages=messages,
                **args,
            )
            results.append(response)

        return [element["message"]["content"] for element in results]


class OptionSelectingByLogProbsInferenceEngine:
    """OptionSelectingByLogProbsInferenceEngine inference engine is used to select an option based on the logprobs of an options list conditioned by a prompt.

    The inference engines that inherit from this class must implement `get_token_count` and `get_options_log_probs`.
    """

    @abc.abstractmethod
    def get_token_count(self, dataset):
        """Get the token count of the source key of each dict of the dataset. Add to each instance in the data a "token_count" field.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries, each representing a data instance.

        Returns:
            List[int]: The token count of the texts
        """

    @abc.abstractmethod
    def get_options_log_probs(self, dataset):
        """Get the token logprobs of the options of the key task_data.options of each dict of the dataset.

        Add to each instance in the data a "options_log_prob" field, which is a dict with str as key and a list of {text: str, logprob:float}.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries, each representing a data instance.

        Returns:
            List[int]: The token count of the texts
        """

    def select(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate most likely labels based on log probabilities for a set of fixed completions."""
        dataset_with_token_counts = self.get_token_count(dataset)
        token_counts = [d["token_count"] for d in dataset_with_token_counts]

        # pass in the token count so we only return the option score
        dataset_with_options = [
            {
                "source": instance["source"] + option,
                "task_data": {"token_count": token_count},
            }
            for instance, token_count in zip(dataset, token_counts)
            for option in instance["task_data"]["options"]
        ]

        dataset_with_options_logprobs: list[
            list[dict[str, float | str]]
        ] = self.get_options_log_probs(dataset_with_options)

        dataset_iterator = iter(dataset_with_options_logprobs)

        for instance in dataset:
            tokens_with_logprob_list = []
            # get the input tokens for the completions of the current resp_idx
            for _ in instance["task_data"]["options"]:
                tokens_with_logprob = next(dataset_iterator)["prediction"]
                tokens_with_logprob_list.append(tokens_with_logprob)
            # we start comparing all the options, e.g. if there are five options the value will be [0,1,2,3,4]
            to_compare_indexes = list(range(len(instance["task_data"]["options"])))
            # token_with_logprob_comp is the logprobs and the text of the tokens
            # for each of the options at a specific index
            for token_with_logprob_comp in zip(*tokens_with_logprob_list):
                tokens_comp = [t["text"] for t in token_with_logprob_comp]
                logprobs_comp = [t["logprob"] for t in token_with_logprob_comp]
                # Find the maximum value by comparing the logprob of the nth token of non-discarded options
                index_max = max(
                    (
                        (val, idx)
                        for idx, val in enumerate(logprobs_comp)
                        if idx in to_compare_indexes
                    ),
                    key=lambda x: x[0],
                )[1]
                # get the token of the biggest logprob
                token_value_with_max_logprob = tokens_comp[index_max]
                # check that the token is not repeated in the non-discarded options
                count = tokens_comp.count(token_value_with_max_logprob)
                if count > 1:
                    # multiple tokens with same max logprob, we need to continue iterating
                    to_compare_indexes = [
                        index
                        for index, token_value in enumerate(tokens_comp)
                        if token_value == token_value_with_max_logprob
                    ]
                    continue
                # we got the index of the maximum log_prob that doesn't have a duplicated token value at other index
                break

            if len(to_compare_indexes) > 1:
                # multiple options are either equal or have the same token values prefix
                # choose the first
                index_max = to_compare_indexes[0]

            instance["prediction"] = instance["task_data"]["options"][index_max]
        return dataset


class IbmGenAiInferenceEngine(
    InferenceEngine,
    IbmGenAiInferenceEngineParamsMixin,
    PackageRequirementsMixin,
    LogProbInferenceEngine,
    OptionSelectingByLogProbsInferenceEngine,
):
    label: str = "ibm_genai"
    model_name: str
    _requirements_list = {
        "ibm-generative-ai": "Install ibm-genai package using 'pip install --upgrade ibm-generative-ai"
    }
    data_classification_policy = ["public", "proprietary"]
    parameters: Optional[IbmGenAiInferenceEngineParams] = None

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

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

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
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

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
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
            return TextGenerationInferenceOutput(
                prediction=predict_result,
                input_tokens=result.input_token_count,
                output_tokens=result.generated_token_count,
                model_name=self.model_name,
                inference_type=self.label,
            )
        return predict_result

    def get_token_count(self, dataset):
        texts = [instance["source"] for instance in dataset]
        token_counts = list(
            tqdm(
                [
                    result.token_count
                    for response in self.client.text.tokenization.create(
                        model_id=self.model_name,
                        input=texts,
                        execution_options={"ordered": True},
                    )
                    for result in response.results
                ],
                desc="Tokenizing",
                total=len(texts),
            )
        )
        for i, token_count in enumerate(token_counts):
            dataset[i]["token_count"] = token_count
        return dataset

    def get_options_log_probs(self, dataset):
        """Add to each instance in the data a "options_log_prob" field, which is a dict with str as key and a list of {text: str, logprob:float}."""
        from genai.schema import TextGenerationParameters, TextGenerationReturnOptions

        texts = [x["source"] for x in dataset]

        responses = tqdm(
            self.client.text.generation.create(
                model_id=self.model_name,
                inputs=texts,
                execution_options={"ordered": True},
                parameters=TextGenerationParameters(
                    max_new_tokens=1,
                    return_options=TextGenerationReturnOptions(
                        input_tokens=True, token_logprobs=True
                    ),
                    # random_seed=self.random_state
                ),
            ),
            total=len(texts),
            desc="Completions",
        )

        scores = [
            [
                {"text": token.text, "logprob": token.logprob}
                for token in response.results[0].input_tokens
            ]
            for response in responses
        ]

        for instance, score in zip(dataset, scores):
            instance["prediction"] = score[instance["task_data"]["token_count"] - 1 :]
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

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

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

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        outputs = []
        for instance in tqdm(dataset, desc="Inferring with openAI API"):
            messages = self.to_messages(instance)
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                **self._get_completion_kwargs(),
            )
            prediction = response.choices[0].message.content
            output = self.get_return_object(prediction, response, return_meta_data)

            outputs.append(output)

        return outputs

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
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
            return TextGenerationInferenceOutput(
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

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

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

    def _infer_chat(self, instance: Dict[str, Any]) -> str:
        messages = self.to_messages(instance)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self._get_infer_kwargs(),
        )
        return response.choices[0].message.content

    def _infer_text(self, instance: Dict[str, Any]) -> str:
        response = self.client.completions.create(
            model=self.model_name,
            prompt=instance["source"],
            **self._get_infer_kwargs(),
        )
        return response.choices[0].text

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        from together.types.models import ModelType

        outputs = []
        if self.model_type == ModelType.CHAT:
            for instance in tqdm(dataset, desc="Inferring with Together AI Chat API"):
                outputs.append(self._infer_chat(instance))
        else:
            self.verify_not_chat_api(dataset)
            for instance in tqdm(dataset, desc="Inferring with Together AI Text API"):
                outputs.append(self._infer_text(instance))
        return outputs


class VLLMRemoteInferenceEngine(OpenAiInferenceEngine):
    label: str = "vllm"

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
    OptionSelectingByLogProbsInferenceEngine,
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
        "ibm-watsonx-ai==1.1.14": "Install ibm-watsonx-ai package using 'pip install --upgrade ibm-watsonx-ai'. "
        "It is advised to have Python version >=3.10 installed, as at lower version this package "
        "may cause conflicts with other installed packages."
    }
    data_classification_policy = ["public", "proprietary"]
    parameters: Optional[WMLInferenceEngineParams] = None
    concurrency_limit: int = 10
    _client: Any = InternalField(default=None, name="WML client")

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

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

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        self.verify_not_chat_api(dataset)
        model, params = self._load_model_and_params()

        result = []
        for source in dataset["source"]:
            instance_result = model.generate(
                prompt=source,
                params=self.to_dict([WMLInferenceEngineParamsMixin], keep_empty=False),
            )
            prediction = instance_result["results"][0]["generated_text"]
            instance_final_results = self.get_return_object(
                prediction, instance_result, return_meta_data
            )
            result.append(instance_final_results)

        return result

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        self.verify_not_chat_api(dataset)

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
            prompt=[instance["source"] for instance in dataset],
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
            return TextGenerationInferenceOutput(
                prediction=predict_result,
                input_tokens=result["results"][0]["input_token_count"],
                output_tokens=result["results"][0]["generated_token_count"],
                model_name=self.model_name,
                inference_type=self.label,
            )
        return predict_result

    def get_token_count(self, dataset):
        from ibm_watsonx_ai.foundation_models import ModelInference

        texts = [instance["source"] for instance in dataset]

        model = ModelInference(
            model_id=self.model_name,
            deployment_id=self.deployment_id,
            api_client=self._client,
        )

        for i in trange(len(texts), desc="Tokenizing"):
            response = model.tokenize(prompt=texts[i], return_tokens=True)["result"]
            dataset[i]["token_count"] = response["token_count"]

        return dataset

    def get_options_log_probs(self, dataset):
        """Add to each instance in the data a "options_log_prob" field, which is a dict with str as key and a list of {text: str, logprob:float}."""
        from ibm_watsonx_ai.foundation_models import ModelInference

        model = ModelInference(
            model_id=self.model_name,
            deployment_id=self.deployment_id,
            api_client=self._client,
        )

        texts = [x["source"] for x in dataset]

        responses = list(
            tqdm(
                model.generate(
                    prompt=texts,
                    params={
                        "decoding_method": "greedy",
                        "max_new_tokens": 1,
                        "return_options": {
                            "input_tokens": True,
                            "token_logprobs": True,
                        },
                    },
                ),
                total=len(texts),
                desc="Completions",
            )
        )

        scores = [
            [
                {
                    "text": token["text"],
                    "logprob": token["logprob"] if "logprob" in token else 1,
                }
                for token in response["results"][0]["input_tokens"]
            ]
            for response in responses
        ]

        for instance, score in zip(dataset, scores):
            instance["prediction"] = score[instance["task_data"]["token_count"] - 1 :]
        return dataset


def get_images_without_text(instance):
    return extract_images(instance["source"], instance)


def get_text_without_images(instance, image_token="<image>"):
    regex = r"<" + f"{constants.image_tag}" + r'\s+src=["\'](.*?)["\']\s*/?>'
    return re.sub(regex, image_token, instance["source"])


class HFLlavaInferenceEngine(InferenceEngine, LazyLoadMixin):
    model_name: str
    max_new_tokens: int
    lazy_load = True
    image_token = "<image>"

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers",
        "torch": "Install torch, go on PyTorch website for mode details.",
        "accelerate": "pip install accelerate",
    }

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, "hf_lava")

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

    def _get_input(self, instance):
        assert isinstance(instance["source"], list), "Must use format=formats.chat_api"
        images = []
        conversation = []
        for turn in instance["source"]:
            if isinstance(turn["content"], list):
                for content in turn["content"]:
                    if content["type"] == "image_url":
                        content["type"] = "image"
                        image_url = content.pop("image_url")["url"]
                        image = data_url_to_image(image_url)
                        images.append(image)
            conversation.append(turn)
        return conversation, images

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if not self._is_loaded():
            self._prepare_engine()

        import torch

        results = []
        for instance in tqdm(dataset):
            conversation, images = self._get_input(instance)

            if len(images) == 1:
                images = images[0]

            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            inputs = self.processor(images=images, text=text, return_tensors="pt").to(
                self.device, torch.float16
            )

            input_len = len(inputs["input_ids"][0])
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            result = self.processor.decode(
                output[0][input_len:], skip_special_tokens=True
            )
            results.append(result)

        return results


class LMMSEvalBaseInferenceEngine(
    InferenceEngine, PackageRequirementsMixin, LazyLoadMixin
):
    model_type: str
    model_args: Dict[str, str]
    batch_size: int = 1
    image_token = "<image>"

    _requirements_list = ["lmms-eval==0.2.4"]

    def prepare_engine(self):
        if not self.lazy_load:
            self._prepare_engine()

    def _prepare_engine(self):
        import torch
        from lmms_eval.api.instance import Instance
        from lmms_eval.models import get_model

        self.new_instance = Instance

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if isinstance(self.model_args, dict):
            self.model_args = ",".join(f"{k}={v}" for k, v in self.model_args.items())

        self.model = get_model(self.model_type).create_from_arg_string(
            self.model_args,
            {
                "batch_size": self.batch_size,
                "device": self.device,
            },
        )

    def _is_loaded(self):
        return hasattr(self, "model") and self.model is not None


class LMMSEvalInferenceEngine(LMMSEvalBaseInferenceEngine):
    max_new_tokens: int = 32
    temperature: float = 0.0
    do_sample: bool = False
    generate_until: List[str] = ["\n\n"]

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        self.verify_not_chat_api(dataset)
        if not self._is_loaded():
            self._prepare_engine()

        from lmms_eval.api.instance import Instance

        temp_task_name = str(uuid.uuid4())

        requests = []
        for i, instance in enumerate(dataset):
            requests.append(
                Instance(
                    request_type="generate_until",
                    arguments=(
                        get_text_without_images(instance, image_token=self.image_token),
                        {
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": self.temperature,
                            "do_sample": self.do_sample,
                            "until": self.generate_until,
                        },
                        get_images_without_text,
                        i,
                        temp_task_name,
                        "test",
                    ),
                    idx=i,
                    metadata={
                        "task": temp_task_name,
                        "doc_id": i,
                        "repeats": 1,
                    },
                )
            )

        self.model.task_dict[temp_task_name] = DatasetDict({"test": dataset})

        responses = self.model.generate_until(requests)

        self.model.task_dict.pop(temp_task_name)

        return responses


class LMMSEvalLoglikelihoodInferenceEngine(LMMSEvalBaseInferenceEngine):
    request_type: Literal["loglikelihood"] = "loglikelihood"

    def make_instance(self, instance, special_args, index, task_name):
        from lmms_eval.api.instance import Instance

        return Instance(
            request_type=self.request_type,
            arguments=(
                get_text_without_images(instance, image_token=self.image_token),
                special_args,
                get_images_without_text,
                index,
                task_name,
                "test",
            ),
            idx=index,
            metadata={
                "task": task_name,
                "doc_id": index,
                "repeats": 1,
            },
        )

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if not self._is_loaded():
            self._prepare_engine()

        temp_task_name = str(uuid.uuid4())

        requests = []
        for i, instance in enumerate(dataset):
            task_data = instance["task_data"]

            if isinstance(task_data, str):
                task_data = json.loads(task_data)

            for option in task_data["options"]:
                requests.append(
                    self.make_instance(
                        instance,
                        option,
                        i,
                        temp_task_name,
                    )
                )

        self.model.task_dict[temp_task_name] = DatasetDict({"test": dataset})
        self.model.metadata = {}

        responses = self.model.loglikelihood(requests)

        self.model.task_dict.pop(temp_task_name)

        optimal_scores = [sys.float_info.max] * len(dataset)
        optimal_responses = [None] * len(dataset)

        for request, (score, _) in zip(requests, responses):
            if score < optimal_scores[request.idx]:
                optimal_scores[request.idx] = score
                optimal_responses[request.idx] = request.arguments[1]

        return optimal_responses


class VLLMInferenceEngine(
    InferenceEngine, PackageRequirementsMixin, StandardAPIParamsMixin
):
    def prepare_engine(self):
        from vllm import LLM, SamplingParams

        args = self.to_dict([StandardAPIParamsMixin])
        self.sampling_params = SamplingParams(**args)
        self.llm = LLM(model=self.model)

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        inputs = []
        for instance in dataset:
            inputs.append(instance["source"])

        if isinstance(inputs[0], list):
            outputs = self.llm.chat(inputs, self.sampling_params)
        else:
            outputs = self.llm.generate(inputs, self.sampling_params)

        predictions = []
        for output in outputs:
            predictions.append(output.outputs[0].text)

        return predictions


class AsyncTokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate  # Tokens added per second
        self.capacity = capacity  # Maximum tokens in the bucket
        self.tokens = capacity
        self.timestamp = time.perf_counter()
        self.lock = asyncio.Lock()
        self.interval = 1.0 / self.rate  # Time between tokens

    async def acquire(self, tokens=1):
        while True:
            async with self.lock:
                now = time.perf_counter()
                delta = now - self.timestamp

                # Calculate the number of tokens to add
                token_intervals = int(delta / self.interval)
                if token_intervals > 0:
                    self.tokens = min(self.capacity, self.tokens + token_intervals)
                    self.timestamp += token_intervals * self.interval
                    logging.debug(
                        f"Added {token_intervals} tokens. Tokens now: {self.tokens}"
                    )

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logging.debug(f"Token acquired. Tokens left: {self.tokens}")
                    return
                # Calculate time until the next token is available
                time_until_next_token = self.interval - (now - self.timestamp)
                logging.debug(
                    f"Not enough tokens. Need to wait {time_until_next_token:.4f} seconds."
                )
            # Sleep outside the lock to allow other coroutines to proceed
            await asyncio.sleep(time_until_next_token)


class LiteLLMInferenceEngine(
    InferenceEngine, StandardAPIParamsMixin, PackageRequirementsMixin
):
    max_requests_per_second: float = 6
    max_retries: int = 5  # Set to 0 to prevent internal retries

    _requirements_list: list = ["litellm", "tenacity", "tqdm", "diskcache"]

    def prepare_engine(self):
        # Initialize the token bucket rate limiter
        self._rate_limiter = AsyncTokenBucket(
            rate=self.max_requests_per_second,
            capacity=self.max_requests_per_second,
        )
        self.inference_type = "litellm"
        import litellm
        from litellm import acompletion
        from litellm.caching.caching import Cache

        litellm.cache = Cache(type="disk")

        self._completion = acompletion
        # Initialize a semaphore to limit concurrency
        self._semaphore = asyncio.Semaphore(self.max_requests_per_second)

    async def _infer_instance(
        self, index: int, instance: Dict[str, Any]
    ) -> TextGenerationInferenceOutput:
        """Process a single inference request."""
        async with self._semaphore:
            await self._rate_limiter.acquire()
            # Introduce a slight delay to prevent burstiness
            await asyncio.sleep(0.01)
            messages = self.to_messages(instance)
            kwargs = self.to_dict([StandardAPIParamsMixin])
            try:
                response = await self._completion(
                    messages=messages,
                    max_retries=self.max_retries,
                    caching=True,
                    **kwargs,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error inferring the following instance:\n{instance}"
                ) from e

            usage = response.get("usage", {})
            return TextGenerationInferenceOutput(
                prediction=response["choices"][0]["message"]["content"],
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                model_name=response.get("model", self.model),
                inference_type=self.inference_type,
            )

    async def _infer_async(
        self, dataset: List[Dict[str, Any]]
    ) -> List[TextGenerationInferenceOutput]:
        """Process multiple inference requests concurrently with a progress bar."""
        tasks = [
            self._infer_instance(i, instance) for i, instance in enumerate(dataset)
        ]
        # Use tqdm_asyncio.gather to display progress bar
        return await tqdm_asyncio.gather(
            *tasks, desc=f"LiteLLM Inference ({self.model})", total=len(tasks)
        )

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], "DatasetDict"],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        """Main inference entry point."""
        loop = asyncio.get_event_loop()
        responses = loop.run_until_complete(self._infer_async(dataset))

        if return_meta_data:
            return responses

        return [response.prediction for response in responses]


_supported_apis = Literal[
    "watsonx", "together-ai", "open-ai", "aws", "ollama", "bam", "watsonx-sdk"
]


class CrossProviderInferenceEngine(InferenceEngine, StandardAPIParamsMixin):
    """Inference engine capable of dynamically switching between multiple providers APIs.

    This class extends the InferenceEngine and OpenAiInferenceEngineParamsMixin
    to enable seamless integration with various API providers. The supported APIs are
    specified in `_supported_apis`, allowing users to interact with multiple models
    from different sources. The `api_model_map` dictionary maps each API to
    specific model identifiers, enabling automatic configuration based on
    user requests.

    Attributes:
        provider: Optional; Specifies the current API in use. Must be one of the
            literals in `_supported_apis`.
        provider_model_map: Dictionary mapping each supported API to a corresponding
            model identifier string. This mapping allows consistent access to models
            across different API backends.
    """

    provider: Optional[_supported_apis] = None

    provider_model_map: Dict[_supported_apis, Dict[str, str]] = {
        "watsonx": {
            "llama-3-8b-instruct": "watsonx/meta-llama/llama-3-8b-instruct",
            "llama-3-70b-instruct": "watsonx/meta-llama/llama-3-70b-instruct",
            "granite-3-8b-instruct": "watsonx/ibm/granite-3-8b-instruct",
            "flan-t5-xxl": "watsonx/google/flan-t5-xxl",
            "llama-3-2-1b-instruct": "watsonx/meta-llama/llama-3-2-1b-instruct",
        },
        "watsonx-sdk": {
            "llama-3-8b-instruct": "meta-llama/llama-3-8b-instruct",
            "llama-3-70b-instruct": "meta-llama/llama-3-70b-instruct",
            "granite-3-8b-instruct": "ibm/granite-3-8b-instruct",
        },
        "together-ai": {
            "llama-3-8b-instruct": "together_ai/togethercomputer/llama-3-8b-instruct",
            "llama-3-70b-instruct": "together_ai/togethercomputer/llama-3-70b-instruct",
            "llama-3-2-1b-instruct": "together_ai/togethercomputer/llama-3-2-1b-instruct",
        },
        "aws": {
            "llama-3-8b-instruct": "bedrock/meta.llama3-8b-instruct-v1:0",
            "llama-3-70b-instruct": "bedrock/meta.llama3-70b-instruct-v1:0",
        },
        "ollama": {
            "llama-3-8b-instruct": "llama3:8b",
            "llama-3-70b-instruct": "llama3:70b",
        },
        "bam": {
            "granite-3-8b-instruct": "ibm/granite-8b-instruct-preview-4k",
            "llama-3-8b-instruct": "meta-llama/llama-3-8b-instruct",
            "llama-3-2-1b-instruct": "meta-llama/llama-3-2-1b-instruct",
            "flan-t5-xxl": "google/flan-t5-xxl",
        },
    }

    _provider_to_base_class = {
        "watsonx": LiteLLMInferenceEngine,
        "open-ai": LiteLLMInferenceEngine,
        "together-ai": LiteLLMInferenceEngine,
        "aws": LiteLLMInferenceEngine,
        "ollama": OllamaInferenceEngine,
        "bam": IbmGenAiInferenceEngine,
        "watsonx-sdk": WMLInferenceEngine,
    }

    _provider_param_renaming = {
        "bam": {"max_tokens": "max_new_tokens", "model": "model_name"},
        "watsonx-sdk": {"max_tokens": "max_new_tokens", "model": "model_name"},
    }

    def get_provider_name(self):
        return self.provider if self.provider is not None else settings.default_provider

    def prepare_engine(self):
        provider = self.get_provider_name()
        if provider not in self._provider_to_base_class:
            raise UnitxtError(
                f"{provider} a known API. Supported apis: {','.join(self.provider_model_map.keys())}"
            )
        if self.model not in self.provider_model_map[provider]:
            raise UnitxtError(
                f"{self.model} is not configured for provider {provider}. Supported models: {','.join(self.provider_model_map[provider].keys())}"
            )
        cls = self.__class__._provider_to_base_class[provider]
        args = self.to_dict([StandardAPIParamsMixin])
        args["model"] = self.provider_model_map[provider][self.model]
        params = list(args.keys())
        if provider in self._provider_param_renaming:
            for param in params:
                if args[param] is not None:
                    if param in self._provider_param_renaming[provider]:
                        args[self._provider_param_renaming[provider][param]] = args[
                            param
                        ]
                        del args[param]
                else:
                    del args[param]
        self.engine = cls(**args)

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return self.engine._infer(dataset, return_meta_data)

    def get_engine_id(self):
        api = self.get_provider_name()
        return get_model_and_label_id(self.provider_model_map[api][self.model], api)


class HFOptionSelectingInferenceEngine(InferenceEngine):
    """HuggingFace based class for inference engines that calculate log probabilities.

    This class uses models from the HuggingFace Transformers library to calculate log probabilities for text inputs.
    """

    model_name: str
    batch_size: int

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers"
    }

    def prepare_engine(self):
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
            if isinstance(texts[0], list):
                batch = self.tokenizer.apply_chat_template(batch, tokenize=False)

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

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        inputs = []

        for instance in dataset:
            for option in instance["task_data"]["options"]:
                if isinstance(instance["source"], list):
                    inputs.append(
                        instance["source"] + [{"role": "assistant", "content": option}]
                    )
                else:
                    inputs.append(instance["source"] + option)

        scores = self.get_log_probs(inputs)

        scores_iterator = iter(scores)

        predictions = []
        for instance in dataset:
            options_scores = Counter()
            for option in instance["task_data"]["options"]:
                score = next(scores_iterator)
                options_scores[option] = score
            predictions.append(options_scores.most_common(1)[0][0])

        return predictions
