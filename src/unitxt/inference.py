import abc
import os
from dataclasses import field
from typing import Any, Dict, Generator, List, Literal, Optional, Union

from tqdm import tqdm

from . import get_logger
from .artifact import Artifact
from .deprecation_utils import deprecation
from .operator import PackageRequirementsMixin, PagedStreamOperator

logger = get_logger()


class InferenceEngine(abc.ABC, PagedStreamOperator):
    """Abstract base class for inference."""

    @abc.abstractmethod
    def _infer(self, dataset):
        """Perform inference on the input dataset."""
        pass

    def _infer_log_probs(self, dataset):
        """Perform inference on the input dataset."""
        raise NotImplementedError(
            f"{self.__name__} does not support inference returning log probabilities"
        )

    def infer(self, dataset):
        """Verifies instances of a dataset and performs inference."""
        for instance in dataset:
            self.verify_instance(instance)
        outputs, _metadata = self._infer(dataset)
        return outputs

    def infer_log_probs(self, dataset):
        """Verifies instances of a dataset and performs inference that returns log probabilities of top tokens.

        For each instance , returns a list of top tokens per position.
        [ "top_tokens": [ "text": ..., "logprob": ...} , ... ]

        """
        for instance in dataset:
            self.verify_instance(instance)
        return self._infer_log_probs(dataset)

    def process(self, page: List[Dict], stream_name: Optional[str] = None) -> Generator:
        for instance, output, metadata in zip(page, *self._infer(page)):
            instance["prediction"] = output
            instance["generation_info"] = metadata
            yield instance


class HFPipelineBasedInferenceEngine(InferenceEngine, PackageRequirementsMixin):
    model_name: str
    max_new_tokens: int
    use_fp16: bool = True
    _requirement = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers"
    }

    def prepare(self):
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

    def _infer(self, dataset):
        outputs = []
        metadata = []
        for output in self.model([instance["source"] for instance in dataset]):
            if isinstance(output, list):
                output = output[0]
            outputs.append(output["generated_text"])
            metadata.append(None)
        return outputs, metadata


class MockInferenceEngine(InferenceEngine):
    model_name: str

    def prepare(self):
        return

    def _infer(self, dataset):
        return ["[[10]]" for instance in dataset]


@deprecation(version="1.11.0", alternative="Plain dictionary")
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
    parameters: Union[IbmGenAiInferenceEngineParams, dict[str, Any]] = field(
        default_factory=dict
    )
    _requirement = {
        "genai": "Install ibm-genai package using 'pip install --upgrade ibm-generative-ai"
    }
    data_classification_policy = ["public", "proprietary"]

    def prepare(self):
        from genai import Client, Credentials
        from genai.schema import TextGenerationParameters

        api_key_env_var_name = "GENAI_KEY"
        api_key = os.environ.get(api_key_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run IbmGenAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )
        credentials = Credentials(api_key=api_key)
        self.client = Client(credentials=credentials)

        if (
            getattr(self.parameters, "type", None)
            == "ibm_gen_ai_inference_engine_params"
        ):
            logger.warning(
                "Artifact IbmGenAiInferenceEngineParams is deprecated for GenAI parameter definition, please use plain "
                "dictionary. This will raise an exception in the future"
            )
            self.parameters = self.parameters.to_dict()
        TextGenerationParameters.model_validate(self.parameters)

    def _infer(self, dataset):
        from tqdm import tqdm

        prompts = [instance["source"] for instance in dataset]

        pb = tqdm(desc="Running text generation", total=len(prompts))
        predictions = []
        metadata = []

        for resp in self.client.text.generation.create(
            inputs=prompts, model_id=self.model_name, parameters=self.parameters
        ):
            for result in resp.results:
                predictions.append(result.generated_text)
                metadata.append(
                    {
                        "meta": resp.model_dump(exclude={"results"}, exclude_none=True),
                        **result.model_dump(
                            exclude={"generated_text"}, exclude_none=True
                        ),
                    }
                )
                pb.update()
        pb.close()
        return predictions, metadata


class OpenAiInferenceEngineParams(Artifact):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = 20


class OpenAiInferenceEngine(InferenceEngine, PackageRequirementsMixin):
    label: str = "openai"
    model_name: str
    parameters: OpenAiInferenceEngineParams = field(
        default_factory=OpenAiInferenceEngineParams
    )
    _requirement = {
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
        metadata = []
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
            metadata.append(response.model_dump(exclude={"choices"}))

        return outputs, metadata

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
