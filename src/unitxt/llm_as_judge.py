from typing import Any, Dict, List, Literal, Optional

import evaluate

from .api import produce
from .inference import InferenceEngine, OpenAiInferenceEngine
from .metrics import BulkInstanceMetric


class LLMAsJudge(BulkInstanceMetric):
    """LLM as judge based metric class for evaluating correctness.

    Attributes:
        main_score (str): The main score label used for evaluation.
        task (Literal["rating.single_turn"]): The type of task the llm-as-judge runs. This defines the output and input format of the jude model.
        template (str): The template used when generating inputs for the judge llm.
        format (str): The format used when generating inputs for judge llm.
        system_prompt (str): The system prompt used when generating inputs for judge llm.
        inference_model (InferenceEngine): the module that creates the inference of the judge llm.
        reduction_map (dict): A dictionary specifying the reduction method for the metric.
        batch_size (int): The size of the bulk.
    """

    main_score: str = "llm_as_judge"
    task: Literal["rating.single_turn"] = "rating.single_turn"
    template: str
    format: Optional[str] = None
    system_prompt: Optional[str] = None
    inference_model: InferenceEngine
    reduction_map: Optional[Dict[str, List[str]]] = None
    batch_size: int = 32

    def prepare(self):
        super().prepare()
        if self.reduction_map is None:
            self.reduction_map = {"mean": [self.main_score]}

        supported_tasks = ["rating.single_turn"]
        assert self.task in supported_tasks, (
            f"Error in 'LLMAsJudge' metric. {self.task} is not a supported task type."
            f"The supported tasks types are: {', '.join(supported_tasks)}."
        )

        if isinstance(self.inference_model, OpenAiInferenceEngine):
            if self.format:
                raise ValueError(
                    "Error in 'LLMAsJudge' metric. Inference model 'OpenAiInferenceEngine' does "
                    "not support formatting. Please remove the format definition from the recipe"
                    " (OpenAi Chat API take care of the formatting automatically)."
                )
            if self.system_prompt:
                raise ValueError(
                    "Error in 'LLMAsJudge' metric. Inference model 'OpenAiInferenceEngine' does "
                    "not support system prompt. Please remove the system_prompt definition from the recipe"
                    " (Current implementation of Unitxt does not support this."
                    " Support will be added in future updates)."
                )

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        if self.task == "rating.single_turn":
            instances = [
                {
                    **task_data_instance,
                    "input": task_data_instance["source"],
                    "output": prediction,
                    "rating": "[[5]]",  # This is a dummy value that is not used in practice
                }
                for task_data_instance, prediction, reference in zip(
                    task_data, predictions, references
                )
            ]
            card = "cards.dynamic_cards_for_llm_judges.rating.single_turn"
            recipe = (
                f"card={card},"
                f"template={self.template},"
                "demos_pool_size=0,"
                "num_demos=0"
            )
            if self.system_prompt:
                recipe = f"{recipe},system_prompt={self.system_prompt}"
            if self.format:
                recipe = f"{recipe},format={self.format}"

            dataset = produce(instances, recipe)
            verdicts = self.inference_model.infer(dataset)
            meta_metric = evaluate.load("unitxt/metric")
            meta_scores = meta_metric.compute(predictions=verdicts, references=dataset)
            return [
                {self.main_score: instance["prediction"]} for instance in meta_scores
            ]

        raise NotImplementedError(
            f"Error in 'LLMAsJudge' metric. {self.task} is not a supported task type."
        )
