import abc

from .artifact import Artifact


class InferenceEngine(abc.ABC, Artifact):
    """Abstract base class for inference."""

    @abc.abstractmethod
    def infer(self, dataset):
        """Perform inference on the input dataset."""
        pass


class PipelineBasedInferenceEngine(Artifact):
    """Abstract base class for inference."""

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
