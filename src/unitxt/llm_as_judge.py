from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .api import evaluate, produce
from .inference import InferenceEngine, OpenAiInferenceEngine
from .metrics import BulkInstanceMetric
from .operator import SequentialOperator


class LLMAsJudge(BulkInstanceMetric, ABC):
    """LLM as judge based metric class for evaluating correctness.

    Attributes:
        main_score (str): The main score label used for evaluation.
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

    @abstractmethod
    def _get_instance_for_judge_model(
        self, input_instances: List[str], predictions: List, references: List
    ) -> List[Dict]:
        pass

    def prepare(self):
        super().prepare()
        if self.reduction_map is None:
            self.reduction_map = {"mean": [self.main_score]}

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
        instances = self._get_instance_for_judge_model(
            input_instances, predictions, references
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


class LLMAsJudgeSingleModelSingleTurn(LLMAsJudge):
    template_model_input_field_name: Optional[str]
    template_model_output_field_name: Optional[str]
    template_reference_field_name: Optional[str]

    def prepare(self):
        super().prepare()
        assert (
            self.template_model_input_field_name
            or self.template_model_output_field_name
            or self.template_reference_field_name
        ), (
            "Error in 'LLMAsJudgeForSingleModelSingleTurn' metric."
            " At least one of the following input fields"
            " must not be 'None':"
            " 'template_model_input_field_label',"
            " 'template_model_output_field_label,"
            " 'template_reference_field_label."
        )
        if self.template_model_input_field_name is None:
            self.template_model_input_field_name = "dummy_input"
        if self.template_model_output_field_name is None:
            self.template_model_output_field_name = "dummy_output"
        if self.template_reference_field_name is None:
            self.template_reference_field_name = "dummy_reference"

    def _get_instance_for_judge_model(
        self, input_instances: List[str], predictions: List, references: List
    ) -> List[Dict]:
        return [
            {
                self.template_model_input_field_name: input_instance,
                self.template_model_output_field_name: prediction,
                self.template_reference_field_name: reference,
                "score": "dummy",  # This is a dummy value that is not used in practice
            }
            for input_instance, prediction, reference in zip(
                input_instances, predictions, references
            )
        ]
