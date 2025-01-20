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
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

from datasets import Dataset, DatasetDict
from tqdm import tqdm, trange
from tqdm.asyncio import tqdm_asyncio

from .artifact import Artifact
from .dataclass import InternalField, NonPositionalField
from .deprecation_utils import deprecation
from .error_utils import UnitxtError
from .image_operators import (
    EncodeImageToString,
    ImageDataString,
    data_url_to_image,
    extract_images,
)
from .logging_utils import get_logger
from .operator import PackageRequirementsMixin
from .operators import ArtifactFetcherMixin
from .settings_utils import get_constants, get_settings
from .type_utils import isoftype

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
    credentials: Optional[Dict[str, str]] = {}
    extra_headers: Optional[Dict[str, str]] = None


class TorchDeviceMixin(Artifact):
    device: Optional[str] = None

    def get_device_id(self) -> str:
        if self.device is not None:
            return self.device

        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def get_device(self):
        import torch

        return torch.device(self.get_device_id())


def get_model_and_label_id(model_name, label):
    model_id = model_name.split("/")[-1].replace("-", "_").replace(".", ",").lower()
    return f"{model_id}_{label}"


@dataclasses.dataclass
class TextGenerationInferenceOutput:
    """Contains the prediction results and metadata for the inference.

    Args:
        prediction (Union[str, List[Dict[str, Any]]]): If this is the result of an _infer call, the string predicted by the model.
        | If this is the results of an _infer_log_probs call, a list of dictionaries. The i'th dictionary represents
          the i'th token in the response. The entry "top_tokens" in the dictionary holds a sorted list of the top tokens
          for this position and their probabilities.
        | For example: ``[ {.. "top_tokens": [ {"text": "a", 'logprob': },  {"text": "b", 'logprob': } ....]},
          {.. "top_tokens": [ {"text": "c", 'logprob': },  {"text": "d", 'logprob': } ....]} ]``

        input_tokens (int) : number of input tokens to the model.

        output_tokens (int) : number of output tokens to the model.

        stop_reason (str): stop reason for text generation, for example "eos" (end of string).

        seed (int): seed used by the model during generation.

        input_text (str): input to the model.

        model_name (str): the model_name as kept in the InferenceEngine.

        inference_type (str): The label stating the type of the InferenceEngine.
    """

    prediction: Union[str, List[Dict[str, Any]]]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    seed: Optional[int] = None
    input_text: Optional[str] = None
    model_name: Optional[str] = None
    inference_type: Optional[str] = None


T = TypeVar("T")


class ListWithMetadata(List[T]):
    def __init__(self, *args, metadata: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"ListWithMetadata(data={super().__repr__()}, metadata={self.metadata})"


class InferenceEngine(Artifact):
    """Abstract base class for inference."""

    @abc.abstractmethod
    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
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

    def __call__(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[ListWithMetadata[str], ListWithMetadata[TextGenerationInferenceOutput]]:
        return self.infer(dataset=dataset, return_meta_data=return_meta_data)

    def infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[ListWithMetadata[str], ListWithMetadata[TextGenerationInferenceOutput]]:
        """Verifies instances of a dataset and perform inference on the input dataset.

        If return_meta_data - returns a list of TextGenerationInferenceOutput, else returns a list of the string
        predictions.
        """
        if not isoftype(dataset, Union[List[Dict[str, Any]], Dataset]):
            raise Exception(
                "Dataset passed to infer() is not list of dictionaries or Huggingface Dataset"
            )
        if return_meta_data and not hasattr(self, "get_return_object"):
            raise NotImplementedError(
                f"Inference engine {self.__class__.__name__} does not support return_meta_data as it "
                f"does not contain a 'get_return_object' method. Please set return_meta_data=False."
            )

        [self.verify_instance(instance) for instance in dataset]
        if settings.mock_inference_mode:
            result = self._mock_infer(dataset)
        else:
            result = self._infer(dataset, return_meta_data)
        return ListWithMetadata(
            result,
            metadata={
                "init_dict": self._init_dict,
                "inference_engine_type": self.__class__.__name__,
                "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            },
        )

    def _mock_infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
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

    def get_model_details(self) -> Dict:
        """Might not be possible to implement for all inference engines. Returns an empty dict by default."""
        return {}

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
        dataset: Union[List[Dict[str, Any]], Dataset],
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
        dataset: Union[List[Dict[str, Any]], Dataset],
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


class HFGenerationParamsMixin(Artifact):
    max_new_tokens: int
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_beams: Optional[int] = None
    repetition_penalty: Optional[float] = None
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class HFInferenceEngineBase(
    InferenceEngine,
    LogProbInferenceEngine,
    PackageRequirementsMixin,
    LazyLoadMixin,
    HFGenerationParamsMixin,
    TorchDeviceMixin,
):
    model_name: str
    label: str

    n_top_tokens: int = 5

    device_map: Any = None

    use_fast_tokenizer: bool = True
    low_cpu_mem_usage: bool = True
    torch_dtype: str = "torch.float16"

    model: Any = InternalField(default=None, name="Inference object")
    processor: Any = InternalField(default=None, name="Input processor (tokenizer)")

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers",
        "torch": "Install torch, go on PyTorch website for mode details.",
        "accelerate": "pip install accelerate",
    }

    def _is_loaded(self):
        return hasattr(self, "model") and self.model is not None

    def _set_inference_device(self):
        if self.device is not None and self.device_map is not None:
            raise ValueError(
                f"You must specify either 'device' or 'device_map', however both "
                f"were given: 'device={self.device}', 'device_map={self.device_map}'."
            )

        if self.device_map is None:
            self.device = self.get_device()

    @abc.abstractmethod
    def _init_processor(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_model(self):
        raise NotImplementedError

    def _get_torch_dtype(self):
        import torch

        if not isinstance(self.torch_dtype, str) or not self.torch_dtype.startswith(
            "torch."
        ):
            raise ValueError(
                f"'torch_dtype' must be a string representing torch data "
                f"type used for inference. The name should be an absolute "
                f"import, for example: 'torch.float16'. However, "
                f"'{self.torch_dtype}' was given instead."
            )

        try:
            dtype = eval(self.torch_dtype)
        except (AttributeError, TypeError) as e:
            raise ValueError(
                f"Incorrect value of 'torch_dtype' was given: '{self.torch_dtype}'."
            ) from e

        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"'torch_dtype' must be an instance of 'torch.dtype', however, "
                f"'{dtype}' is an instance of '{type(dtype)}'."
            )

        return dtype

    def _prepare_engine(self):
        self._set_inference_device()
        self._init_processor()
        self._init_model()

    def prepare_engine(self):
        if not self.lazy_load:
            self._prepare_engine()

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

    def decode_tokens(self, tokens: Sequence, inp_length: int) -> List[str]:
        return [
            self.processor.decode(token, skip_special_tokens=True)
            for token in tokens[inp_length:]
        ]

    @staticmethod
    def create_string_from_tokens(string_tokens: List[str]) -> str:
        return "".join(token for token in string_tokens)

    def make_predictions(self, prepared_inputs: Mapping) -> Mapping:
        return self.model.generate(
            **prepared_inputs,
            **self.to_dict([HFGenerationParamsMixin], keep_empty=False),
            output_scores=True,
            return_dict_in_generate=True,
        )

    def compute_transition_scores(
        self, sequences: Sequence, scores: Sequence, beam_indices: Optional[int]
    ) -> Sequence:
        # Some models may not support computing scores in this form by default, so a possible
        # child class should have its own implementation of this method if necessary.
        return self.model.compute_transition_scores(
            sequences,
            scores,
            normalize_logits=True,
            beam_indices=beam_indices,
        )

    def get_logprobs(
        self, predictions: Mapping, string_tokens: List[List[str]]
    ) -> List[List[Dict[str, Any]]]:
        beam_indices = (
            predictions.beam_indices
            if self.num_beams is not None and self.num_beams > 1
            else None
        )

        transition_scores = self.compute_transition_scores(
            sequences=predictions.sequences,
            scores=predictions.scores,
            beam_indices=beam_indices,
        )

        logprobs: List[List[Dict[str, Any]]] = []

        for sample_no, sample_scores in enumerate(transition_scores.detach().cpu()):
            sample_logprobs: List[Dict[str, Any]] = []

            for n, score in enumerate(sample_scores):
                sample_logprobs.append(
                    {
                        "text": string_tokens[sample_no][n],
                        "logprob": float(score.cpu()),
                        "top_tokens": [
                            {
                                "text": self.processor.decode(idx),
                                "logprob": float(
                                    predictions.scores[n][sample_no][idx].cpu()
                                ),
                            }
                            for idx in predictions.scores[n][sample_no].argsort(
                                dim=0, descending=True
                            )[: self.n_top_tokens]
                        ],
                    }
                )

            logprobs.append(sample_logprobs)

        return logprobs

    @abc.abstractmethod
    def prepare_inputs(self, data: Iterable) -> Mapping:
        raise NotImplementedError

    def get_return_object(
        self,
        output: Union[str, List[Dict[str, Any]]],
        output_tokens: Optional[int],
        inp: Optional[str],
        inp_tokens: Optional[int],
        return_meta_data: bool,
    ) -> Union[str, List[Dict[str, Any]], TextGenerationInferenceOutput]:
        if return_meta_data:
            return TextGenerationInferenceOutput(
                prediction=output,
                output_tokens=output_tokens if output_tokens is not None else None,
                input_text=inp,
                input_tokens=inp_tokens if inp_tokens is not None else None,
                model_name=self.model_name,
                inference_type=self.label,
            )
        return output

    def infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if not self._is_loaded():
            self._prepare_engine()
        return super().infer(dataset, return_meta_data)

    @abc.abstractmethod
    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        raise NotImplementedError

    def infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        if not self._is_loaded():
            self._prepare_engine()
        return super().infer_log_probs(dataset, return_meta_data)

    @abc.abstractmethod
    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        raise NotImplementedError


class HFAutoModelInferenceEngine(HFInferenceEngineBase):
    label: str = "hf_auto_model"

    def _init_processor(self):
        from transformers import AutoTokenizer

        self.processor = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            use_fast=self.use_fast_tokenizer,
            padding=True,
            truncation=True,
        )

    def _init_model(self):
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        )

        model_class = (
            AutoModelForSeq2SeqLM
            if AutoConfig.from_pretrained(self.model_name).is_encoder_decoder
            else AutoModelForCausalLM
        )

        self.model = model_class.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            trust_remote_code=True,
            device_map=self.device_map,
            torch_dtype=self._get_torch_dtype(),
        )
        if self.device_map is None:
            self.model.to(self.device)

    def prepare_inputs(self, data: Iterable) -> Mapping:
        if isinstance(data[0], list):
            data = self.processor.apply_chat_template(
                data, tokenize=False, add_generation_prompt=True
            )
        return self.processor(
            data,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device or self.device_map)

    def _infer_fn(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool,
        return_logprobs: bool,
    ) -> Union[List[str], List[Dict], List[TextGenerationInferenceOutput]]:
        tokenized_inputs = self.prepare_inputs(
            [instance["source"] for instance in dataset]
        )
        input_length = (
            1
            if self.model.config.is_encoder_decoder
            else tokenized_inputs.input_ids.shape[1]
        )

        predictions = self.make_predictions(tokenized_inputs)
        sequences = predictions.sequences

        string_tokens = [
            self.decode_tokens(sequence, input_length) for sequence in sequences
        ]

        final_outputs = (
            self.get_logprobs(predictions, string_tokens)
            if return_logprobs
            else [self.create_string_from_tokens(strings) for strings in string_tokens]
        )

        return [
            self.get_return_object(
                output=final_outputs[i],
                output_tokens=len(string_tokens[i]),
                inp=dataset[i]["source"],
                inp_tokens=len(tokenized_inputs.encodings[i].tokens)
                if tokenized_inputs.encodings is not None
                else None,
                return_meta_data=return_meta_data,
            )
            for i in range(len(sequences))
        ]

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return self._infer_fn(dataset, return_meta_data, False)

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        self.verify_not_chat_api(dataset)
        return self._infer_fn(dataset, return_meta_data, True)


class HFLlavaInferenceEngine(HFInferenceEngineBase):
    lazy_load: bool = True
    label: str = "hf_lava"
    image_token: str = "<image>"

    def compute_transition_scores(
        self, sequences: Sequence, scores: Sequence, beam_indices: Optional[int]
    ) -> Sequence:
        if not hasattr(self.model.config, "vocab_size"):
            self.model.config.vocab_size = self.model.vocab_size

        return super().compute_transition_scores(sequences, scores, beam_indices)

    def _init_processor(self):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        if not self.pad_token_id and hasattr(self.processor, "eos_token_id"):
            self.pad_token_id = self.processor.eos_token_id

    def _init_model(self):
        from transformers import LlavaForConditionalGeneration

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self._get_torch_dtype(),
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            device_map=self.device_map,
        )
        if self.device_map is None:
            self.model.to(self.device)

    @staticmethod
    def _get_input(instance):
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

    def prepare_inputs(self, data: Iterable) -> Mapping:
        conversation, images = self._get_input(data)

        if len(images) == 1:
            images = images[0]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs: Mapping = self.processor(
            images=images, text=text, return_tensors="pt"
        ).to(self.device or self.device_map, self._get_torch_dtype())

        return inputs

    def _infer_fn(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool,
        return_logprobs: bool,
    ) -> Union[List[str], List[Dict], List[TextGenerationInferenceOutput]]:
        results = []

        for instance in tqdm(dataset):
            processed_inputs = self.prepare_inputs(instance)
            input_len = len(processed_inputs["input_ids"][0])

            predictions = self.make_predictions(processed_inputs)

            string_tokens = self.decode_tokens(predictions.sequences[0], input_len)

            final_outputs = (
                self.get_logprobs(predictions, [string_tokens])[0]
                if return_logprobs
                else self.create_string_from_tokens(string_tokens)
            )

            results.append(
                self.get_return_object(
                    output=final_outputs,
                    output_tokens=len(string_tokens),
                    inp=instance["source"],
                    inp_tokens=None,
                    return_meta_data=return_meta_data,
                )
            )

        return results

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return self._infer_fn(dataset, return_meta_data, False)

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        return self._infer_fn(dataset, return_meta_data, True)


class HFPeftInferenceEngine(HFAutoModelInferenceEngine):
    label: str = "hf_peft_auto_model"

    peft_config: Any = InternalField(
        default=None,
        name="PEFT config read from the directory or the Hub repository "
        "id specified in the 'model_name'.",
    )

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers",
        "torch": "Install torch, go on PyTorch website for mode details.",
        "accelerate": "pip install accelerate",
        "peft": "Install 'peft' package using: 'pip install peft'.",
    }

    def _prepare_engine(self):
        self._read_peft_config()
        super()._prepare_engine()

    def _read_peft_config(self):
        from peft import PeftConfig

        try:
            config = PeftConfig.from_pretrained(self.model_name)
            assert isinstance(config.base_model_name_or_path, str)
            self.peft_config = config

        except ValueError as e:
            if "Can't find" in str(e):
                raise ValueError(
                    f"Specified model '{self.model_name}' is not the PEFT model. "
                    f"Use a regular instance of the `HFAutoModelInferenceEngine` "
                    f"instead."
                ) from e

            raise e

    def _init_processor(self):
        from transformers import AutoTokenizer

        self.processor = AutoTokenizer.from_pretrained(
            self.peft_config.base_model_name_or_path
        )

    def _init_model(self):
        from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM
        from transformers import AutoConfig

        model_class = (
            AutoPeftModelForSeq2SeqLM
            if AutoConfig.from_pretrained(self.model_name).is_encoder_decoder
            else AutoPeftModelForCausalLM
        )

        self.model = model_class.from_pretrained(
            pretrained_model_name_or_path=self.peft_config.base_model_name_or_path,
            trust_remote_code=True,
            device_map=self.device_map,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            torch_dtype=self._get_torch_dtype(),
        )
        if self.device_map is None:
            self.model.to(self.device)


class HFPipelineBasedInferenceEngine(
    InferenceEngine,
    PackageRequirementsMixin,
    LazyLoadMixin,
    HFGenerationParamsMixin,
    TorchDeviceMixin,
):
    model_name: str
    label: str = "hf_pipeline_inference_engine"

    use_fast_tokenizer: bool = True
    use_fp16: bool = True
    load_in_8bit: bool = False

    task: Optional[str] = None

    device_map: Any = None

    pipe: Any = InternalField(default=None)

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers",
        "torch": "Install torch, go on PyTorch website for mode details.",
        "accelerate": "pip install accelerate",
    }

    def _is_loaded(self):
        return hasattr(self, "model") and self.model is not None

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, "hf_pipeline")

    def _define_task(self):
        from transformers import AutoConfig

        self.task = (
            "text2text-generation"
            if AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            ).is_encoder_decoder
            else "text-generation"
        )

    def _get_model_args(self) -> Dict[str, Any]:
        import torch
        from transformers import BitsAndBytesConfig

        args = {}

        if self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=self.load_in_8bit)
            args["quantization_config"] = quantization_config
        elif self.use_fp16:
            if self.device == torch.device("mps"):
                args["torch_dtype"] = torch.float16
            else:
                args["torch_dtype"] = torch.bfloat16

        # We do this, because in some cases, using device:auto will offload some weights to the cpu
        # (even though the model might *just* fit to a single gpu), even if there is a gpu available, and this will
        # cause an error because the data is always on the gpu
        if torch.cuda.device_count() > 1:
            assert self.device == torch.device(0)
            args["device_map"] = "auto"
        else:
            if not self.load_in_8bit:
                args["device"] = self.device

        if self.task == "text-generation":
            args["return_full_text"] = False

        return args

    def _create_pipeline(self, model_args: Dict[str, Any]):
        from transformers import pipeline

        self.model = pipeline(
            model=self.model_name,
            task=self.task,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=True,
            **model_args,
            **self.to_dict(
                [HFGenerationParamsMixin],
                keep_empty=False,
            ),
        )

    def _set_inference_device(self):
        if self.device is not None and self.device_map is not None:
            raise ValueError(
                f"You must specify either 'device' or 'device_map', however both "
                f"were given: 'device={self.device}', 'device_map={self.device_map}'."
            )

        if self.device_map is None:
            self.device = self.get_device()

    def _prepare_engine(self):
        self._set_inference_device()
        if self.task is None:
            self._define_task()
        model_args = self._get_model_args()
        self._create_pipeline(model_args)

    def prepare_engine(self):
        if not self.lazy_load:
            self._prepare_engine()

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if not self._is_loaded():
            self._prepare_engine()

        outputs = self.model([instance["source"] for instance in dataset])

        return [
            self.get_return_object(output[0], instance["source"], return_meta_data)
            if isinstance(output, list)
            else self.get_return_object(output, instance["source"], return_meta_data)
            for output, instance in zip(outputs, dataset)
        ]

    def get_return_object(self, output, inp, return_meta_data):
        if return_meta_data:
            return TextGenerationInferenceOutput(
                prediction=output["generated_text"],
                model_name=self.model_name,
                inference_type=self.label,
                input_text=inp,
            )
        return output["generated_text"]


def mock_logprobs_default_value_factory() -> List[Dict[str, Any]]:
    return [
        {
            "logprob": -1,
            "text": "[[10]]",
            "top_tokens": [
                {"logprob": -1, "text": "[[10]]"},
            ],
        }
    ]


class MockInferenceEngine(InferenceEngine, LogProbInferenceEngine):
    model_name: str
    default_inference_value: str = "[[10]]"
    default_inference_value_logprob: List[Dict[str, Any]] = dataclasses.field(
        default_factory=mock_logprobs_default_value_factory,
    )
    label: str = "mock_inference_engine"

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, "mock")

    def prepare_engine(self):
        return

    def _mock_infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return [self.default_inference_value for _ in dataset]

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return [
            self.get_return_object(
                self.default_inference_value, instance, return_meta_data
            )
            for instance in dataset
        ]

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        return [
            self.get_return_object(
                self.default_inference_value_logprob, instance, return_meta_data
            )
            for instance in dataset
        ]

    def get_return_object(self, predict_result, instance, return_meta_data):
        if return_meta_data:
            return TextGenerationInferenceOutput(
                prediction=predict_result,
                input_tokens=len(instance["source"]),
                output_tokens=len(predict_result),
                model_name=self.model_name,
                inference_type=self.label,
                input_text=instance["source"],
                seed=111,
                stop_reason="",
            )
        return predict_result


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


class GenericInferenceEngine(
    InferenceEngine, ArtifactFetcherMixin, LogProbInferenceEngine
):
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
        # If mock_inference_mode is set, no engine is prepared.
        if hasattr(self, "engine"):
            return f"generic_{self.engine.get_engine_id()}"
        return "generic_inference_engine"

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return self.engine._infer(dataset)

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if not isinstance(self.engine, LogProbInferenceEngine):
            raise NotImplementedError(
                f"Error in infer: inference engine used by the GenericInferenceEngine"
                f"({self.engine.__class__.__name__}) does not support logprobs."
            )
        return self.engine._infer_log_probs(dataset)


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
        dataset: Union[List[Dict[str, Any]], Dataset],
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

        dataset_with_options_logprobs: List[
            List[Dict[str, Union[float, str]]]
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
    rate_limit: int = 10

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

    @staticmethod
    def _get_credentials():
        from genai import Credentials

        api_key_env_var_name = "GENAI_KEY"
        api_key = os.environ.get(api_key_env_var_name)

        assert api_key is not None, (
            f"Error while trying to run IbmGenAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )

        return Credentials(api_key=api_key)

    def prepare_engine(self):
        self.check_missing_requirements()

        from genai import Client
        from genai.text.generation import CreateExecutionOptions

        credentials = self._get_credentials()
        self.client = Client(credentials=credentials)

        self.execution_options = CreateExecutionOptions(
            concurrency_limit=self.rate_limit
        )

        self._set_inference_parameters()

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        from genai.schema import TextGenerationParameters, TextGenerationResult

        self.verify_not_chat_api(dataset)

        genai_params = TextGenerationParameters(
            **self.to_dict([IbmGenAiInferenceEngineParamsMixin])
        )

        responses = self.client.text.generation.create(
            model_id=self.model_name,
            inputs=[instance["source"] for instance in dataset],
            parameters=genai_params,
            execution_options=self.execution_options,
        )

        results = []
        for response in responses:
            generation_result: TextGenerationResult = response.results[0]
            result = self.get_return_object(
                generation_result.generated_text, generation_result, return_meta_data
            )
            results.append(result)
        return results

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        from genai.schema import TextGenerationParameters, TextGenerationResult

        self.verify_not_chat_api(dataset)

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
            execution_options=self.execution_options,
        )

        predict_results = []
        for prediction in predictions:
            result: TextGenerationResult = prediction.results[0]
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
                input_text=result.input_text,
                seed=self.random_seed,
                stop_reason=result.stop_reason,
            )
        return predict_result

    def get_model_details(self) -> Dict:
        from genai import ApiClient
        from genai.model import ModelService

        api_client = ApiClient(credentials=self._get_credentials())
        model_info = (
            ModelService(api_client=api_client).retrieve(id=self.model_name).result
        )
        return model_info.dict()

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


class CredentialsOpenAi(TypedDict, total=False):
    api_key: str
    api_url: str


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


def run_with_imap(func):
    def inner(self, args):
        return func(self, *args)

    return inner


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
    base_url: Optional[str] = None
    default_headers: Dict[str, str] = {}
    credentials: CredentialsOpenAi = {}
    num_parallel_requests: int = 20

    def get_engine_id(self) -> str:
        return get_model_and_label_id(self.model_name, self.label)

    def _prepare_credentials(self) -> CredentialsOpenAi:
        api_key = self.credentials.get(
            "api_key", os.environ.get(f"{self.label.upper()}_API_KEY", None)
        )
        assert api_key, (
            f"Error while trying to run {self.label}. "
            f"Please set the env variable: '{self.label.upper()}_API_KEY'"
        )

        api_url = self.credentials.get(
            "api_url", os.environ.get(f"{self.label.upper()}_API_URL", None)
        )

        return {"api_key": api_key, "api_url": api_url}

    def get_default_headers(self) -> Dict[str, str]:
        return self.default_headers

    def create_client(self):
        from openai import OpenAI

        self.credentials = self._prepare_credentials()
        return OpenAI(
            api_key=self.credentials["api_key"],
            base_url=self.base_url or self.credentials["api_url"],
            default_headers=self.get_default_headers(),
        )

    def prepare_engine(self):
        self.client = self.create_client()
        self._set_inference_parameters()

    def _get_completion_kwargs(self):
        return {
            k: v
            for k, v in self.to_dict([OpenAiInferenceEngineParamsMixin]).items()
            if v is not None
        }

    def _parallel_infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        infer_func,
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        inputs = [(instance, return_meta_data) for instance in dataset]
        outputs = []
        with ThreadPool(processes=self.num_parallel_requests) as pool:
            for output in tqdm(
                pool.imap(infer_func, inputs),
                total=len(inputs),
                desc=f"Inferring with {self.__class__.__name__}",
            ):
                outputs.append(output)

        return outputs

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return self._parallel_infer(
            dataset=dataset,
            return_meta_data=return_meta_data,
            infer_func=self._get_chat_completion,
        )

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        return self._parallel_infer(
            dataset=dataset,
            return_meta_data=return_meta_data,
            infer_func=self._get_logprobs,
        )

    @run_with_imap
    def _get_chat_completion(self, instance, return_meta_data):
        messages = self.to_messages(instance)
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            **self._get_completion_kwargs(),
        )
        prediction = response.choices[0].message.content
        return self.get_return_object(prediction, response, return_meta_data)

    @run_with_imap
    def _get_logprobs(self, instance, return_meta_data):
        messages = self.to_messages(instance)
        response = self.client.chat.completions.create(
            messages=messages,
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
        return self.get_return_object(pred_output, response, return_meta_data)

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


class AzureOpenAIInferenceEngine(OpenAiInferenceEngine):
    label: str = "azure_openai"

    def _prepare_credentials(self) -> CredentialsOpenAi:
        api_key_var_name = f"{self.label.upper()}_API_KEY"
        api_key = self.credentials.get(
            "api_key", os.environ.get(api_key_var_name, None)
        )
        assert api_key, (
            f"Error while trying to run {self.label}. "
            f"Please set the env variable: '{api_key_var_name}'"
        )

        azure_openapi_host = self.credentials.get(
            "azure_openapi_host", os.environ.get(f"{self.label.upper()}_HOST", None)
        )

        api_version = self.credentials.get(
            "api_version", os.environ.get("OPENAI_API_VERSION", None)
        )
        assert (
            api_version and azure_openapi_host
        ), "Error while trying to run AzureOpenAIInferenceEngine: Missing environment variable param AZURE_OPENAI_HOST or OPENAI_API_VERSION"
        api_url = f"{azure_openapi_host}/openai/deployments/{self.model_name}/chat/completions?api-version={api_version}"

        return {"api_key": api_key, "api_url": api_url}

    def create_client(self):
        from openai import AzureOpenAI

        self.credentials = self._prepare_credentials()
        return AzureOpenAI(
            api_key=self.credentials["api_key"],
            base_url=self.credentials["api_url"],
            default_headers=self.get_default_headers(),
        )


class VLLMRemoteInferenceEngine(OpenAiInferenceEngine):
    label: str = "vllm"


class RITSInferenceEngine(
    OpenAiInferenceEngine,
):
    label: str = "rits"
    data_classification_policy = ["public", "proprietary"]

    def get_default_headers(self):
        return {"RITS_API_KEY": self.credentials["api_key"]}

    def prepare_engine(self):
        # inference endpoint need the '/v1' path
        self.base_url = (
            RITSInferenceEngine.get_base_url_from_model_name(self.model_name) + "/v1"
        )
        logger.info(f"Created RITS inference engine with base url: {self.base_url}")
        super().prepare_engine()

    @staticmethod
    def get_base_url_from_model_name(model_name: str):
        base_url_template = (
            "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{}"
        )
        return base_url_template.format(
            RITSInferenceEngine._get_model_name_for_endpoint(model_name)
        )

    @staticmethod
    def _get_model_name_for_endpoint(model_name: str):
        return (
            model_name.split("/")[-1]
            .lower()
            .replace("v0.1", "v01")
            .replace("vision-", "")
            .replace(".", "-")
        )


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
        dataset: Union[List[Dict[str, Any]], Dataset],
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


@deprecation(
    version="2.0.0",
    msg=" You can specify inference parameters directly when initializing an inference engine.",
)
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


class WMLGenerationParamsMixin(Artifact):
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


class WMLChatParamsMixin(Artifact):
    frequency_penalty: Optional[float] = None
    top_logprobs: Optional[int] = 5
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    time_limit: Optional[int] = None
    top_p: Optional[float] = None
    n: Optional[int] = None


CredentialsWML = Dict[
    Literal["url", "username", "password", "apikey", "project_id", "space_id"], str
]


class WMLInferenceEngineBase(
    InferenceEngine,
    PackageRequirementsMixin,
    LogProbInferenceEngine,
    OptionSelectingByLogProbsInferenceEngine,
):
    """Base for classes running inference using ibm-watsonx-ai.

    Args:
        credentials (Dict[str, str], optional):
            By default, it is created by a class
            instance which tries to retrieve proper environment variables
            ("WML_URL", "WML_PROJECT_ID", "WML_SPACE_ID", "WML_APIKEY", "WML_USERNAME", "WML_PASSWORD").
            However, a dictionary with the following keys: "url", "apikey", "project_id", "space_id",
            "username", "password".
            can be directly provided instead.
        model_name (str, optional):
            ID of a model to be used for inference. Mutually
            exclusive with 'deployment_id'.
        deployment_id (str, optional):
            Deployment ID of a tuned model to be used for
            inference. Mutually exclusive with 'model_name'.
        parameters (Union[WMLInferenceEngineParams, WMLGenerationParamsMixin, WMLChatParamsMixin], optional):
            Defines inference parameters and their values. Deprecated attribute, please pass respective
            parameters directly to the respective class instead.
    """

    credentials: Optional[CredentialsWML] = None
    model_name: Optional[str] = None
    deployment_id: Optional[str] = None
    label: str = "wml"
    _requirements_list = {
        "ibm_watsonx_ai": "Install ibm-watsonx-ai package using 'pip install --upgrade ibm-watsonx-ai'. "
        "It is advised to have Python version >=3.10 installed, as at lower version this package "
        "may cause conflicts with other installed packages."
    }
    data_classification_policy = ["public", "proprietary"]
    parameters: Optional[
        Union[WMLInferenceEngineParams, WMLGenerationParamsMixin, WMLChatParamsMixin]
    ] = None

    _client: Any = InternalField(default=None, name="WML client")
    _model: Any = InternalField(default=None, name="WML model")

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name or self.deployment_id, self.label)

    def verify(self):
        super().verify()

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

    def _initialize_wml_client(self):
        from ibm_watsonx_ai.client import APIClient

        if self.credentials is None:
            self.credentials = self._read_wml_credentials_from_env()
        self._verify_wml_credentials(self.credentials)

        client = APIClient(credentials=self.credentials)
        if "space_id" in self.credentials:
            client.set.default_space(self.credentials["space_id"])
        else:
            client.set.default_project(self.credentials["project_id"])
        return client

    @staticmethod
    def _read_wml_credentials_from_env() -> CredentialsWML:
        credentials: CredentialsWML = {}

        url = os.environ.get("WML_URL")
        assert url, (
            "Error while trying to run 'WMLInferenceEngine'. "
            "Please set the env variable: 'WML_URL'"
        )
        credentials["url"] = url

        space_id = os.environ.get("WML_SPACE_ID")
        project_id = os.environ.get("WML_PROJECT_ID")
        if space_id and project_id:
            get_logger().warning(
                "Either 'WML_SPACE_ID' or 'WML_PROJECT_ID' need to be "
                "specified, however, both were found. 'WMLInferenceEngine' "
                "will use space by default. If it is not desired, then have "
                "only one of those defined in the env."
            )
            credentials["space_id"] = space_id
        elif project_id:
            credentials["project_id"] = project_id
        else:
            raise AssertionError(
                "Error while trying to run 'WMLInferenceEngine'. "
                "Please set either 'WML_SPACE_ID' or 'WML_PROJECT_ID' env "
                "variable."
            )

        apikey = os.environ.get("WML_APIKEY")
        username = os.environ.get("WML_USERNAME")
        password = os.environ.get("WML_PASSWORD")

        if apikey and username and password:
            get_logger().warning(
                "Either 'WML_APIKEY' or both 'WML_USERNAME' and 'WML_PASSWORD' "
                "need to be specified, however, all of them were found. "
                "'WMLInferenceEngine' will use api key only by default. If it is not "
                "desired, then have only one of those options defined in the env."
            )

        if apikey:
            credentials["apikey"] = apikey
        elif username and password:
            credentials["username"] = username
            credentials["password"] = password
        else:
            raise AssertionError(
                "Error while trying to run 'WMLInferenceEngine'. "
                "Please set either 'WML_APIKEY' or both 'WML_USERNAME' and "
                "'WML_PASSWORD' env variables."
            )

        return credentials

    @staticmethod
    def _verify_wml_credentials(credentials: CredentialsWML) -> None:
        assert isoftype(credentials, CredentialsWML), (
            "WML credentials object must be a dictionary which may "
            "contain only the following keys: "
            "['url', 'apikey', 'username', 'password']."
        )

        assert credentials.get(
            "url"
        ), "'url' is a mandatory key for WML credentials dict."
        assert "space_id" in credentials or "project_id" in credentials, (
            "Either 'space_id' or 'project_id' must be provided "
            "as keys for WML credentials dict."
        )
        assert "apikey" in credentials or (
            "username" in credentials and "password" in credentials
        ), (
            "Either 'apikey' or both 'username' and 'password' must be provided "
            "as keys for WML credentials dict."
        )

    def prepare_engine(self):
        self.check_missing_requirements()

        self._client = self._initialize_wml_client()

        self._set_inference_parameters()

    def _load_model(self):
        from ibm_watsonx_ai.foundation_models.inference import ModelInference

        self._model = ModelInference(
            model_id=self.model_name,
            deployment_id=self.deployment_id,
            api_client=self._client,
        )

    @abc.abstractmethod
    def _send_requests(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_logprobs: bool,
        return_meta_data: bool,
    ) -> Union[List[str], List[Dict], List[TextGenerationInferenceOutput]]:
        raise NotImplementedError(
            f"The class '{self.get_pretty_print_name()}' is an abstract class. "
            f"Please used either 'WMLInferenceEngineGeneration' or "
            f"'WMLInferenceEngineChat' instead, depending on your task."
        )

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if self._model is None:
            self._load_model()

        return self._send_requests(
            dataset=dataset,
            return_logprobs=False,
            return_meta_data=return_meta_data,
        )

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        if self._model is None:
            self._load_model()

        return self._send_requests(
            dataset=dataset,
            return_logprobs=True,
            return_meta_data=return_meta_data,
        )

    @abc.abstractmethod
    def get_return_object(self, predict_result, result, input_text, return_meta_data):
        raise NotImplementedError

    def get_model_details(self) -> Dict:
        return self._model.get_details()

    def get_token_count(self, dataset):
        if self._model is None:
            self._load_model()

        texts = [instance["source"] for instance in dataset]

        for i in trange(len(texts), desc="Tokenizing"):
            response = self._model.tokenize(prompt=texts[i], return_tokens=True)[
                "result"
            ]
            dataset[i]["token_count"] = response["token_count"]

        return dataset

    def get_options_log_probs(self, dataset):
        """Add to each instance in the data a "options_log_prob" field, which is a dict with str as key and a list of {text: str, logprob:float}."""
        if self._model is None:
            self._load_model()

        texts = [x["source"] for x in dataset]

        responses = list(
            tqdm(
                self._model.generate(
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


class WMLInferenceEngineGeneration(WMLInferenceEngineBase, WMLGenerationParamsMixin):
    """Generates text for textual inputs.

    If you want to include images in your input, please use 'WMLInferenceEngineChat' instead.

    Args:
        concurrency_limit (int):
            Number of concurrent requests sent to a model. Default is 10,
            which is also the maximum value.

    Examples:
        .. code-block:: python

            from .api import load_dataset

            wml_credentials = {
                "url": "some_url", "project_id": "some_id", "api_key": "some_key"
            }
            model_name = "google/flan-t5-xxl"
            wml_inference = WMLInferenceEngineGeneration(
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

    concurrency_limit: int = 10

    def verify(self):
        super().verify()

        assert (
            isinstance(self.concurrency_limit, int)
            and 1 <= self.concurrency_limit <= 10
        ), (
            f"'concurrency_limit' must be a positive integer not greater than 10. "
            f"However, '{self.concurrency_limit}' was given."
        )

    def _set_logprobs_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        user_return_options = params.pop("return_options", {})
        # currently this is the only configuration that returns generated
        # logprobs and behaves as expected
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

        return {
            **params,
            "return_options": logprobs_return_options,
        }

    def _send_requests(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_logprobs: bool,
        return_meta_data: bool,
    ) -> Union[List[str], List[Dict], List[TextGenerationInferenceOutput]]:
        self.verify_not_chat_api(dataset)

        params = self.to_dict([WMLGenerationParamsMixin], keep_empty=False)

        if return_logprobs:
            generation_type = "generated_tokens"
            params = self._set_logprobs_params(params)
        else:
            generation_type = "generated_text"

        inputs: List[str] = [instance["source"] for instance in dataset]

        results = self._model.generate(
            prompt=inputs,
            params=params,
            concurrency_limit=self.concurrency_limit,
        )

        final_results = []
        for result, inp in zip(results, inputs):
            result_metadata = result["results"][0]
            generated_content = result_metadata[generation_type]
            final_results.append(
                self.get_return_object(
                    generated_content, result_metadata, inp, return_meta_data
                )
            )
        return final_results

    def get_return_object(self, predict_result, result, input_text, return_meta_data):
        if return_meta_data:
            return TextGenerationInferenceOutput(
                prediction=predict_result,
                input_tokens=result["input_token_count"],
                output_tokens=result["generated_token_count"],
                model_name=self.model_name or self.deployment_id,
                inference_type=self.label,
                stop_reason=result["stop_reason"],
                seed=self.random_seed,
                input_text=input_text,
            )
        return predict_result


class WMLInferenceEngineChat(WMLInferenceEngineBase, WMLChatParamsMixin):
    """Creates chat session and returns a model's response.

    You can also include images in your inputs. If you use only textual input, it is
    recommended to use 'WMLInferenceEngineGeneration' instead as it is faster, and allows
    more parameters for text generation.

    You can provide either already formatted messages, or a raw dataset as an input.
    In case of the former, all passed images should be base64-encoded strings given as
    an 'image_url' within a message. Moreover, only one image per a list of messages
    may be sent.
    As for the latter, if there are multiple images per one instance, they will be sent
    separately with the same query. If that could possibly affect expected responses,
    concatenate images within an instance into a single image and adjust your query
    accordingly (if necessary).

    Args:
        image_encoder (EncodeImageToString, optional):
            operator which encodes images in
            given format to base64 strings required by service. You should specify it when
            you are using images in your inputs.

    Example:
        .. code-block:: python

            from .api import load_dataset
            from .image_operators

            image_encoder = EncodeImageToString(image_format="JPEG")

            wml_credentials = {
                "url": "some_url", "project_id": "some_id", "api_key": "some_key"
            }
            model_name = "meta-llama/llama-3-2-11b-vision-instruct"
            wml_inference = WMLInferenceEngineChat(
                credentials=wml_credentials,
                model_name=model_name,
                image_encoder=image_encoder,
                data_classification_policy=["public"],
                max_tokens=1024,
            )

            dataset = load_dataset(
                dataset_query="card=cards.doc_vqa.en,template=templates.qa.with_context.with_type,loader_limit=30"
            )
            results = wml_inference.infer(dataset["test"])
    """

    image_encoder: Optional[EncodeImageToString] = None

    @staticmethod
    def _extract_queries(instance: Dict[str, Any]) -> Tuple[Optional[str], List]:
        task_data = instance["task_data"]
        if isinstance(task_data, str):
            task_data = json.loads(task_data)
        question = task_data.get("question")

        images = [None]
        if "images" in instance["media"]:
            images = extract_images(instance)

        return question or instance["source"], images

    def _create_messages_from_instance(
        self, instance: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        """Method creates chat messages to be sent to a watsonx.ai model based on a given instance from a dataset."""
        text, images = self._extract_queries(instance)

        messages: List[List[Dict[str, Any]]] = []
        base_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }

        # Iteration over all possible images to create a separate message for
        # every single image, since SDK allows only one image per request.
        for image in images:
            message = base_message.copy()

            if image is not None:
                encoded_image = image
                if not isinstance(encoded_image, str):
                    if self.image_encoder is None:
                        raise ValueError(
                            "If sending image queries as well, and they are not "
                            "already encoded to base64 strings, you must specify "
                            "the 'image_encoder' to be used."
                        )
                    encoded_image = self.image_encoder.encode_image_to_base64(image)

                message["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": ImageDataString(
                                "data:image/jpeg;base64," + encoded_image
                            ),
                        },
                    }
                )

            messages.append([message])

        return messages

    @staticmethod
    def verify_messages(messages: List[Dict[str, Any]]):
        """Method verifies if externally provided messages containing images are compatible with the format required by ibm-watsonx-ai."""
        n_images = 0
        for message in messages:
            if isinstance(message["content"], str):
                continue

            for content in message["content"]:
                if isinstance(content, dict):
                    if "image" in content["type"] and content["type"] != "image_url":
                        raise ValueError(
                            f"ibm-watsonx-ai only supports sending images as base64-encoded "
                            f"strings, which should be given as 'image_url' in a message. "
                            f"However, '{content['type']}' was given."
                        )

                    if content["type"] == "image_url":
                        n_images += 1
                    if n_images > 1:
                        raise ValueError(
                            "ibm-watsonx-ai only supports sending one image per a list "
                            "of messages."
                        )

    def to_messages(self, instance: Union[Dict, List]) -> List[List[Dict[str, Any]]]:
        if isinstance(instance["source"], str) and "media" in instance:
            return self._create_messages_from_instance(instance)

        messages = super().to_messages(instance)
        self.verify_messages(messages)
        # This is done to be compatible with inputs containing
        # images as SDK allows sending only one image per message.
        return [messages]

    def _send_requests(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_logprobs: bool,
        return_meta_data: bool,
    ) -> Union[List[str], List[Dict], List[TextGenerationInferenceOutput]]:
        params = self.to_dict([WMLChatParamsMixin], keep_empty=False)

        if return_logprobs:
            output_type = "logprobs"
            params["logprobs"] = True
        else:
            output_type = "message"
            params["logprobs"] = False

        final_results = []

        for instance in dataset:
            messages = self.to_messages(instance)

            for message in messages:
                result = self._model.chat(
                    messages=message,
                    params=params,
                )

                final_results.append(
                    self.get_return_object(
                        result["choices"][0][output_type]["content"],
                        result,
                        instance["source"],
                        return_meta_data,
                    )
                )

        return final_results

    def get_return_object(self, predict_result, result, input_text, return_meta_data):
        if return_meta_data:
            return TextGenerationInferenceOutput(
                prediction=predict_result,
                input_tokens=result["usage"]["prompt_tokens"],
                output_tokens=len(predict_result)
                if isinstance(predict_result, list)
                else None,
                model_name=self.model_name or self.deployment_id,
                inference_type=self.label,
                stop_reason=result["choices"][0]["finish_reason"],
                input_text=input_text,
            )
        return predict_result


@deprecation(
    version="2.0.0",
    msg=" Please use either 'WMLInferenceEngineGeneration' or 'WMLInferenceEngineChat'"
    " depending on your task.",
)
class WMLInferenceEngine(WMLInferenceEngineGeneration):
    def prepare_engine(self):
        super().prepare_engine()
        get_logger().warning("'WMLInferenceEngine' is deprecated")


def get_images_without_text(instance):
    if isinstance(instance["source"], str):
        images = extract_images(instance["source"], instance)
    elif isinstance(instance["source"], list):
        images = []
        for turn in instance["source"]:
            content = turn["content"]
            if isinstance(content, list):
                for sub_content in content:
                    if sub_content["type"] == "image_url":
                        image = data_url_to_image(sub_content["image_url"]["url"])
                        images.append(image)

    return [image.convert("RGB") for image in images]


def get_text_without_images(instance, image_token="<image>"):
    if isinstance(instance["source"], str):
        regex = r"<" + f"{constants.image_tag}" + r'\s+src=["\'](.*?)["\']\s*/?>'
        return re.sub(regex, image_token, instance["source"])
    if isinstance(instance["source"], list):
        text = ""
        for turn in instance["source"]:
            content = turn["content"]
            if isinstance(content, str):
                text += content
            else:
                for sub_content in content:
                    if sub_content["type"] == "text":
                        text += sub_content["text"]
                    if sub_content["type"].startswith("image"):
                        text += image_token
        return text
    raise ValueError()


class LMMSEvalBaseInferenceEngine(
    InferenceEngine, PackageRequirementsMixin, LazyLoadMixin, TorchDeviceMixin
):
    model_type: str
    model_args: Dict[str, str]
    batch_size: int = 1
    image_token = "<image>"

    _requirements_list = {
        "lmms_eval": "Install llms-eval package using 'pip install lmms-eval==0.2.4'",
    }

    def prepare_engine(self):
        if not self.lazy_load:
            self._prepare_engine()

    def _prepare_engine(self):
        from lmms_eval.api.instance import Instance
        from lmms_eval.models import get_model

        self.new_instance = Instance

        self.device = self.get_device()

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
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
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
        dataset: Union[List[Dict[str, Any]], Dataset],
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


class VLLMParamsMixin(Artifact):
    model: str
    n: int = 1
    best_of: Optional[int] = None
    _real_n: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    bad_words: Optional[List[str]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None


class VLLMInferenceEngine(InferenceEngine, PackageRequirementsMixin, VLLMParamsMixin):
    def prepare_engine(self):
        from vllm import LLM, SamplingParams

        args = self.to_dict([VLLMParamsMixin])
        args.pop("model")

        self.sampling_params = SamplingParams(**args)
        self.llm = LLM(model=self.model, trust_remote_code=True)

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
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
    label: str = "litellm"
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
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            del kwargs["credentials"]
            try:
                response = await self._completion(
                    messages=messages,
                    max_retries=self.max_retries,
                    caching=True,
                    drop_params=False,
                    **self.credentials,
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
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        """Main inference entry point."""
        loop = asyncio.get_event_loop()
        responses = loop.run_until_complete(self._infer_async(dataset))
        return self.get_return_object(responses, return_meta_data)

    def get_return_object(self, responses, return_meta_data):
        if return_meta_data:
            return responses

        return [response.prediction for response in responses]


_supported_apis = Literal[
    "watsonx",
    "together-ai",
    "open-ai",
    "aws",
    "ollama",
    "bam",
    "watsonx-sdk",
    "rits",
    "azure",
    "vertex-ai",
]


class CrossProviderInferenceEngine(InferenceEngine, StandardAPIParamsMixin):
    """Inference engine capable of dynamically switching between multiple providers APIs.

    This class extends the InferenceEngine and OpenAiInferenceEngineParamsMixin
    to enable seamless integration with various API providers. The supported APIs are
    specified in ``_supported_apis``, allowing users to interact with multiple models
    from different sources. The ``provider_model_map`` dictionary maps each API to
    specific model identifiers, enabling automatic configuration based on
    user requests.

    Current _supported_apis = ["watsonx", "together-ai", "open-ai", "aws", "ollama",
    "bam", "watsonx-sdk", "rits", "vertex-ai"]

    Args:
        provider (Optional):
            Specifies the current API in use. Must be one of the
            literals in `_supported_apis`.
        provider_model_map (Dict[_supported_apis, Dict[str, str]]):
            mapping each supported API to a corresponding
            model identifier string. This mapping allows consistent access to models
            across different API backends.
    """

    label: str = "cross_provider"
    provider: Optional[_supported_apis] = None

    provider_model_map: Dict[_supported_apis, Dict[str, str]] = {
        "watsonx": {
            "llama-3-8b-instruct": "watsonx/meta-llama/llama-3-8b-instruct",
            "llama-3-70b-instruct": "watsonx/meta-llama/llama-3-70b-instruct",
            "llama-3-1-70b-instruct": "watsonx/meta-llama/llama-3-1-70b-instruct",
            "llama-3-3-70b-instruct": "watsonx/meta-llama/llama-3-3-70b-instruct",
            "granite-3-8b-instruct": "watsonx/ibm/granite-3-8b-instruct",
            "flan-t5-xxl": "watsonx/google/flan-t5-xxl",
            "llama-3-2-1b-instruct": "watsonx/meta-llama/llama-3-2-1b-instruct",
            "llama-3-2-11b-vision-instruct": "watsonx/meta-llama/llama-3-2-11b-vision-instruct",
            "llama-3-2-90b-vision-instruct": "watsonx/meta-llama/llama-3-2-90b-vision-instruct",
        },
        "watsonx-sdk": {
            "llama-3-8b-instruct": "meta-llama/llama-3-8b-instruct",
            "llama-3-70b-instruct": "meta-llama/llama-3-70b-instruct",
            "granite-3-8b-instruct": "ibm/granite-3-8b-instruct",
        },
        "together-ai": {
            "llama-3-8b-instruct": "together_ai/meta-llama/Llama-3-8b-chat-hf",
            "llama-3-70b-instruct": "together_ai/meta-llama/Llama-3-70b-chat-hf",
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
        "rits": {
            "granite-3-8b-instruct": "ibm-granite/granite-3.0-8b-instruct",
            "llama-3-1-8b-instruct": "meta-llama/llama-3-1-8b-instruct",
            "llama-3-1-70b-instruct": "meta-llama/llama-3-1-70b-instruct",
            "llama-3-2-11b-vision-instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "llama-3-2-90b-vision-instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "llama-3-3-70b-instruct": "meta-llama/llama-3-3-70b-instruct",
            "llama-3-1-405b-instruct-fp8": "meta-llama/llama-3-1-405b-instruct-fp8",
            "mistral-large-instruct": "mistralai/mistral-large-instruct-2407",
            "mixtral-8x7b-instruct": "mistralai/mixtral-8x7B-instruct-v0.1",
        },
        "open-ai": {
            "o1-mini": "o1-mini",
            "o1-preview": "o1-preview",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
            "gpt-4o": "gpt-4o",
            "gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
            "gpt-4-turbo-preview": "gpt-4-0125-preview",
            "gpt-4-turbo": "gpt-4-turbo",
            "gpt-4-0125-preview": "gpt-4-0125-preview",
            "gpt-4-1106-preview": "gpt-4-1106-preview",
            "gpt-3.5-turbo-1106": "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301": "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613": "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613": "gpt-3.5-turbo-16k-0613",
            "gpt-4": "gpt-4",
            "gpt-4-0314": "gpt-4-0314",
            "gpt-4-0613": "gpt-4-0613",
            "gpt-4-32k": "gpt-4-32k",
            "gpt-4-32k-0314": "gpt-4-32k-0314",
            "gpt-4-32k-0613": "gpt-4-32k-0613",
            "gpt-4-vision-preview": "gpt-4-vision-preview",
        },
        "azure": {
            "o1-mini": "azure/o1-mini",
            "o1-preview": "azure/o1-preview",
            "gpt-4o-mini": "azure/gpt-4o-mini",
            "gpt-4o": "azure/gpt-4o",
            "gpt-4": "azure/gpt-4",
            "gpt-4-0314": "azure/gpt-4-0314",
            "gpt-4-0613": "azure/gpt-4-0613",
            "gpt-4-32k": "azure/gpt-4-32k",
            "gpt-4-32k-0314": "azure/gpt-4-32k-0314",
            "gpt-4-32k-0613": "azure/gpt-4-32k-0613",
            "gpt-4-1106-preview": "azure/gpt-4-1106-preview",
            "gpt-4-0125-preview": "azure/gpt-4-0125-preview",
            "gpt-4-turbo": "azure/gpt-4-turbo-2024-04-09",
            "gpt-3.5-turbo": "azure/gpt-3.5-turbo",
            "gpt-3.5-turbo-0301": "azure/gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613": "azure/gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k": "azure/gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613": "azure/gpt-3.5-turbo-16k-0613",
            "gpt-4-vision": "azure/gpt-4-vision",
        },
        "vertex-ai": {
            "llama-3-1-8b-instruct": "vertex_ai/meta/llama-3.1-8b-instruct-maas",
            "llama-3-1-70b-instruct": "vertex_ai/meta/llama-3.1-70b-instruct-maas",
            "llama-3-1-405b-instruct": "vertex_ai/meta/llama-3.1-405b-instruct-maas",
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
        "rits": RITSInferenceEngine,
        "azure": LiteLLMInferenceEngine,
        "vertex-ai": LiteLLMInferenceEngine,
    }

    _provider_param_renaming = {
        "bam": {"max_tokens": "max_new_tokens", "model": "model_name"},
        "watsonx-sdk": {"max_tokens": "max_new_tokens", "model": "model_name"},
        "rits": {"model": "model_name"},
    }

    def get_return_object(self, **kwargs):
        return self.engine.get_return_object(kwargs)

    def get_provider_name(self):
        return self.provider if self.provider is not None else settings.default_provider

    def prepare_engine(self):
        provider = self.get_provider_name()
        if provider not in self._provider_to_base_class:
            raise UnitxtError(
                f"{provider} is not a configured API for CrossProviderInferenceEngine. Supported apis: {','.join(self.provider_model_map.keys())}"
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
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return self.engine._infer(dataset, return_meta_data)

    def get_engine_id(self):
        api = self.get_provider_name()
        return get_model_and_label_id(self.provider_model_map[api][self.model], api)


class HFOptionSelectingInferenceEngine(InferenceEngine, TorchDeviceMixin):
    """HuggingFace based class for inference engines that calculate log probabilities.

    This class uses models from the HuggingFace Transformers library to calculate log probabilities for text inputs.
    """

    model_name: str
    batch_size: int

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers"
    }

    def prepare_engine(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = self.get_device()

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
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if return_meta_data and not hasattr(self.engine, "get_return_object"):
            raise NotImplementedError(
                f"Inference engine {self.engine.__class__.__name__} does not support return_meta_data as it "
                f"does not contain a 'get_return_object' method. Please set return_meta_data=False."
            )

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
