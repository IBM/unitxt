"""Inference Engine Module.

This module defines a set of abstract base classes for creating custom inference engines in the context of natural language processing (NLP) or other machine learning tasks. The classes provided in this module are designed to be extended by developers to create engines that can perform specific types of inference, such as text generation, scoring, option selection, or log probability calculation.

Classes:
    - InferenceEngine: A base class for all inference engines. It provides the fundamental framework for processing and inferring data from datasets. This class defines an abstract method `_infer_dataset` that must be implemented by subclasses to define specific inference behavior.

    - TextGenerationInferenceEngine: An abstract base class for inference engines that generate text-based outputs from input datasets. This class inherits from `InferenceEngine` and requires the implementation of the `generate` method, which is used to produce the text generation results.

    - ScoringInferenceEngine: An abstract class designed for inference engines that assign scores to text inputs. It inherits from `InferenceEngine` and mandates the implementation of the `score` method, which should return a list of scored instances.

    - OptionSelectingInferenceEngine: An abstract class for inference engines that select the best option from a set of options for each input instance. This class also inherits from `InferenceEngine` and requires the implementation of the `select` method, which should return the selected options.

    - LogProbInferenceEngine: An abstract base class for inference engines that perform inference and return log probabilities of the top tokens for each position in the text. This class inherits from `Artifact` and requires the implementation of the `_infer_log_probs` method, which performs the actual log probability calculations.
"""
import json
from abc import abstractmethod
from typing import Any, Dict, List

from .artifact import Artifact
from .deprecation_utils import deprecation
from .logging_utils import get_logger


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
