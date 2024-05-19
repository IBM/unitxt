import abc
import os
from dataclasses import field
from typing import Any, Dict, List, Literal, Optional, Union

from .artifact import Artifact
from .operator import PackageRequirementsMixin
from .settings_utils import get_settings


class InferenceEngine(abc.ABC, Artifact):
    """Abstract base class for inference."""

    @abc.abstractmethod
    def infer(self, dataset):
        """Perform inference on the input dataset."""
        pass

    @staticmethod
    def _assert_allow_passing_data_to_remote_api(remote_api_label: str):
        assert get_settings().allow_passing_data_to_remote_api, (
            f"LlmAsJudge metric cannot run send data to remote APIs ({remote_api_label}) when"
            f" unitxt.settings.allow_passing_data_to_remote_api=False."
            f" Set UNITXT_ALLOW_PASSING_DATA_TO_REMOTE_API environment variable, if you want to allow this. "
        )


class HFPipelineBasedInferenceEngine(InferenceEngine, PackageRequirementsMixin):
    model_name: str
    max_new_tokens: int
    use_fp16: bool = True
    _requirement = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers"
    }

    def prepare(self):
        import torch
        from transformers import pipeline

        model_args: Dict[str, Any] = (
            {"torch_dtype": torch.float16} if self.use_fp16 else {}
        )

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

        model_args.update({"max_new_tokens": self.max_new_tokens})

        self.model = pipeline(
            model=self.model_name, trust_remote_code=True, **model_args
        )

    def infer(self, dataset):
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

    def infer(self, dataset):
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
    _requirement = {
        "genai": "Install ibm-genai package using 'pip install --upgrade ibm-generative-ai"
    }

    def prepare(self):
        from genai import Client, Credentials

        api_key_env_var_name = "GENAI_KEY"
        api_key = os.environ.get(api_key_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run IbmGenAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )
        api_endpoint = os.environ.get("GENAI_KEY")
        credentials = Credentials(api_key=api_key, api_endpoint=api_endpoint)
        self.client = Client(credentials=credentials)

        self._assert_allow_passing_data_to_remote_api(self.label)

    def infer(self, dataset):
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

        return list(
            self.client.text.generation.create(
                model_id=self.model_name,
                inputs=[instance["source"] for instance in dataset],
                parameters=genai_params,
            )
        )


class OpenAiInferenceEngineParams(Artifact):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class OpenAiInferenceEngine(InferenceEngine, PackageRequirementsMixin):
    label: str = "openai"
    model_name: str
    parameters: OpenAiInferenceEngineParams = field(
        default_factory=OpenAiInferenceEngineParams
    )
    _requirement = {
        "openai": "Install openai package using 'pip install --upgrade openai"
    }

    def prepare(self):
        from openai import OpenAI

        api_key_env_var_name = "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run OpenAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )

        self.client = OpenAI(api_key=api_key)
        self._assert_allow_passing_data_to_remote_api(self.label)

    def infer(self, dataset):
        return [
            self.client.chat.completions.create(
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
            for instance in dataset
        ]
