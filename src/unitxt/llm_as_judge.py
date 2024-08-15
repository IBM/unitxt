from typing import Any, Dict, List, Literal, Optional

from .api import evaluate, produce
from .artifact import Artifact, fetch_artifact, settings
from .formats import Format
from .inference import InferenceEngine, OpenAiInferenceEngine
from .metrics import BulkInstanceMetric
from .operator import SequentialOperator
from .system_prompts import SystemPrompt
from .templates import Template


class LLMAsJudge(BulkInstanceMetric):
    """LLM as judge based metric class for evaluating correctness.

    Attributes:
        main_score (str): The main score label used for evaluation.
        task (Literal["rating.single_turn"]): The type of task the llm-as-judge runs. This defines the output and input
         format of the jude model.
        template (Template): The template used when generating inputs for the judge llm.
        format (Format): The format used when generating inputs for judge llm.
        system_prompt (SystemPrompt): The system prompt used when generating inputs for judge llm.
        strip_system_prompt_and_format_from_inputs (bool): Whether to strip the system prompt and formatting from the
         inputs that the models that is being judges received, when they are inserted to the llm-as-judge prompt.
        inference_model (InferenceEngine): the module that creates the inference of the judge llm.
        reduction_map (dict): A dictionary specifying the reduction method for the metric.
        batch_size (int): The size of the bulk.
    """

    main_score: str = "llm_as_judge"
    task: Literal[
        "rating.single_turn",
        "rating.single_turn_with_reference",
        "pairwise_comparative_rating.single_turn",
    ]
    template: Template
    format: Format = None
    system_prompt: SystemPrompt = None
    strip_system_prompt_and_format_from_inputs: bool = True
    inference_model: InferenceEngine
    reduction_map: Optional[Dict[str, List[str]]] = None
    batch_size: int = 32
    prediction_type = Any  # Because handled with multiple tasks

    def _get_input_instances(self, task_data: List[Dict]) -> List:
        if self.strip_system_prompt_and_format_from_inputs:
            instances = []
            for task_data_instance in task_data:
                template = task_data_instance["metadata"]["template"]
                template, _ = fetch_artifact(template)
                instance = SequentialOperator(
                    steps=[template, "formats.empty"]
                ).process_instance(
                    {
                        "input_fields": task_data_instance,
                        "reference_fields": task_data_instance,
                    }
                )
                instances.append(instance["source"])
                """
                We also have access to: instance["target"]
                                        instance["references"]
                """
            return instances
        return [t["source"] for t in task_data]

    def _get_instance_for_judge_model(
        self, input_instances: List[str], predictions: List, references: List
    ) -> List[Dict]:
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
        elif self.task == "rating.single_turn_with_reference":
            instances = [
                {
                    "question": input_instance,
                    "answer": prediction,
                    "reference_answer": reference[0],
                    "rating": 5.0,  # This is a dummy value that is not used in practice
                }
                for input_instance, prediction, reference in zip(
                    input_instances, predictions, references
                )
            ]
        elif self.task == "pairwise_comparative_rating.single_turn":
            instances = [
                {
                    "question": input_instance,
                    "answer_a": prediction,
                    "answer_b": reference[0],
                    "model_a": "input_model",
                    "model_b": "baseline_model",
                    "answer_a_preference": 0,  # This is a dummy value that is not used in practice,
                }
                for input_instance, prediction, reference in zip(
                    input_instances, predictions, references
                )
            ]
        else:
            raise NotImplementedError(
                f"Error in 'LLMAsJudge' metric. {self.task} is not a supported task type."
            )
        return instances

    @staticmethod
    def _add_metadata_to_judge_instances(
        instances: List[List[Any]], task_data: List[Dict]
    ):
        for instance, data in zip(instances, task_data):
            instance["data_classification_policy"] = data["metadata"][
                "data_classification_policy"
            ]

    def prepare(self):
        super().prepare()
        if self.task == "pairwise_comparative_rating.single_turn":
            self.reduction_map = {"weighted_win_rate": [self.main_score]}
        if self.reduction_map is None:
            self.reduction_map = {"mean": [self.main_score]}

    def verify(self):
        supported_tasks = [
            "rating.single_turn",
            "rating.single_turn_with_reference",
            "pairwise_comparative_rating.single_turn",
        ]
        assert self.task in supported_tasks, (
            f"Error in 'LLMAsJudge' metric. {self.task} is not a supported task type."
            f"The supported tasks types are: {', '.join(supported_tasks)}."
        )

        if not isinstance(self.template, Template):
            raise ValueError(
                f"Provided template argument to 'LLMAsJudge' metric is not of type Template, but {type(self.template)}"
            )
        if self.format and not isinstance(self.format, Format):
            raise ValueError(
                f"Provided format argument to 'LLMAsJudge' metric is not of type Format, but {type(self.format)}"
            )

        if self.system_prompt and not isinstance(self.system_prompt, SystemPrompt):
            raise ValueError(
                f"Provided system_prompt argument to 'LLMAsJudge' metric is not of type SystemPrompt, but {type(self.system_prompt)}"
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
        instances = self._get_instance_for_judge_model(
            input_instances, predictions, references
        )
        self._add_metadata_to_judge_instances(instances, task_data)

        card = f"cards.dynamic_cards_for_llm_judges.{self.task}"
        recipe_args = {
            "card": card,
            "template": self.template,
            "demos_pool_size": 0,
            "num_demos": 0,
            "__type__": settings.default_recipe,
        }
        if self.system_prompt:
            recipe_args["system_prompt"] = self.system_prompt
        if self.format:
            recipe_args["format"] = self.format
        recipe = Artifact.from_dict(recipe_args)
        dataset = produce(instances, recipe)
        verdicts = self.inference_model.infer(dataset)
        meta_scores = evaluate(predictions=verdicts, data=dataset)

        res_list = []
        for instance, verdict in zip(meta_scores, verdicts):
            if self.task == "pairwise_comparative_rating.single_turn":
                is_model_b_the_baseline = (
                    instance["task_data"]["model_b"] == "baseline_model"
                )
                if is_model_b_the_baseline:
                    model_a_preference_score = instance["processed_prediction"]
                else:
                    model_a_preference_score = instance["processed_prediction"] * -1

                res = {
                    self.main_score: model_a_preference_score,
                    "judge_raw_output": verdict,
                    "judge_raw_input": instance["source"],
                }
            else:
                res = {
                    self.main_score: instance["processed_prediction"],
                    "judge_raw_output": verdict,
                    "judge_raw_input": instance["source"],
                }
            res_list.append(res)

        return res_list
