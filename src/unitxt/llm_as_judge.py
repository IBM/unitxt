from typing import Any, Dict, List, Literal, Optional

from .api import evaluate, produce
from .inference import InferenceEngine, OpenAiInferenceEngine
from .metrics import BulkInstanceMetric
from .operator import SequentialOperator


class LLMAsJudge(BulkInstanceMetric):
    """LLM as judge based metric class for evaluating correctness.

    Attributes:
        main_score (str): The main score label used for evaluation.
        task (Literal["rating.single_turn"]): The type of task the llm-as-judge runs. This defines the output and input
         format of the jude model.
        template (str): The template used when generating inputs for the judge llm.
        format (str): The format used when generating inputs for judge llm.
        system_prompt (str): The system prompt used when generating inputs for judge llm.
        strip_system_prompt_and_format_from_inputs (bool): Whether to strip the system prompt and formatting from the
         inputs that the models that is being judges received, when they are inserted to the llm-as-judge prompt.
        inference_model (InferenceEngine): the module that creates the inference of the judge llm.
        reduction_map (dict): A dictionary specifying the reduction method for the metric.
        batch_size (int): The size of the bulk.
    """

    main_score: str = "llm_as_judge"
    task: Literal["rating.single_turn"] = "rating.single_turn"
    template: str
    format: Optional[str] = None
    system_prompt: Optional[str] = None
    strip_system_prompt_and_format_from_inputs: bool = True
    inference_model: InferenceEngine
    reduction_map: Optional[Dict[str, List[str]]] = None
    batch_size: int = 32

    def _get_input_instances(self, task_data: List[Dict]) -> List:
        if self.strip_system_prompt_and_format_from_inputs:
            instances = []
            for task_data_instance in task_data:
                template = task_data_instance["metadata"]["template"]
                instance = SequentialOperator(
                    steps=[template, "formats.empty"]
                ).process_instance(
                    {"inputs": task_data_instance, "outputs": task_data_instance}
                )
                instances.append(instance["source"])
                """
                We also have access to: instance["target"]
                                        instance["references"]
                """
            return instances
        return [t["source"] for t in task_data]

    def _get_input_instances_old(self, task_data: List[Dict]) -> List:
        instances = []
        for task_data_instance in task_data:
            recipe_for_judge_input_dict = task_data_instance["recipe_metadata"]
            card = recipe_for_judge_input_dict["card"]
            template = recipe_for_judge_input_dict["template"]
            if self.strip_system_prompt_and_format_from_inputs:
                system_prompt = "system_prompts.empty"
                format = "formats.empty"
            else:
                system_prompt = recipe_for_judge_input_dict["system_prompt"]
                format = recipe_for_judge_input_dict["format"]
            recipe = (
                f"card={card},"
                f"template={template},"
                f"system_prompt={system_prompt},"
                f"format={format}"
            )
            instance = produce(task_data_instance, recipe)
            instances.append(instance)
        return instances

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
        input_instances = self._get_input_instances(task_data)
        if self.task == "rating.single_turn":
            instances = [
                {
                    "question": input_instance,
                    "answer": prediction,
                    "rating": 5.0,  # This is a dummy value that is not used in practice
                }
                for input_instance, prediction, reference in zip(
                    input_instances, predictions, references
                )
            ]
        else:
            raise NotImplementedError(
                f"Error in 'LLMAsJudge' metric. {self.task} is not a supported task type."
            )

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
        meta_scores = evaluate(predictions=verdicts, data=dataset)
        return [{self.main_score: instance["prediction"]} for instance in meta_scores]
