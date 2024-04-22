import abc
import os
from dataclasses import dataclass
from typing import List, Optional, Union

from .artifact import Artifact
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


class HFPipelineBasedInferenceEngine(InferenceEngine):
    model_name: str
    max_new_tokens: int

    def prepare(self):
        from transformers import pipeline

        self.model = pipeline(model=self.model_name)

    def infer(self, dataset):
        return [
            output["generated_text"]
            for output in self.model(
                [instance["source"] for instance in dataset],
                max_new_tokens=self.max_new_tokens,
            )
        ]


@dataclass()
class IbmGenAiInferenceEngineParams:
    decoding_method: str = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stop_sequences: Optional[list[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    typical_p: Optional[float] = None


class IbmGenAiInferenceEngine(InferenceEngine):
    label: str = "ibm_genai"
    model_name: str
    parameters: IbmGenAiInferenceEngineParams = IbmGenAiInferenceEngineParams()

    def prepare(self):
        try:
            from genai import Client, Credentials
        except ImportError as e:
            raise ImportError(
                "Failed to import ibm-genai package. "
                "Please run 'pip install --upgrade ibm-generative-ai'"
            ) from e

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

        genai_params = TextGenerationParameters(**self.parameters.__dict__)
        return list(
            self.client.text.generation.create(
                model_id=self.model_name,
                inputs=[instance["source"] for instance in dataset],
                parameters=genai_params,
            )
        )


@dataclass
class OpenAiInferenceEngineParams:
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class OpenAiInferenceEngine(InferenceEngine):
    label: str = "openai"
    model_name: str
    parameters: OpenAiInferenceEngineParams = OpenAiInferenceEngineParams()

    def prepare(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Failed to import openai package. "
                "Please run 'pip install --upgrade openai'"
            ) from e

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
