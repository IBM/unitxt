from typing import Any, Dict, List

import evaluate

from . import produce
from .inference import InferenceEngine
from .metrics import BulkInstanceMetric


class LLMAsJudge(BulkInstanceMetric):
    """LLM as judge based metric class for evaluating correctness.

    Attributes:
        main_score (str): The main score used for evaluation.
        reduction_map (dict): A dictionary specifying the reduction method for the metric.
        betch_size (int): The size of the bulk.
        recipe (str): The unitxt recipe that will be used to create the judge dataset.
        inference (InferenceEngine): the module that creates the inference.

    Methods:
        prepare(self): Initialization method for the metric.
        compute(self, references, predictions, additional_inputs): Method to compute the metric.

    Usage:
        metric = LlamaIndexCorrectnessMetric()
        scores = metric.compute(references, prediction, additional_inputs)
    """

    main_score: str = "llm_as_judge"
    reduction_map: Dict[str, List[str]] = {"mean": [main_score]}
    batch_size: int = 32
    recipe: str
    inference_model: InferenceEngine

    def prepare(self):
        super().prepare()

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        instances = [
            {
                **task_data_instance,
                **{"model_output": prediction, "rating_label": "[[5]]"},
            }
            for task_data_instance, prediction in zip(task_data, predictions)
        ]

        dataset = produce(instances, self.recipe)
        verdicts = self.inference_model.infer(dataset)
        meta_metric = evaluate.load("unitxt/metric")
        meta_scores = meta_metric.compute(predictions=verdicts, references=dataset)
        return [{self.main_score: instance["prediction"]} for instance in meta_scores]
