from typing import Any, Dict, List, Literal, Optional

from .api import infer
from .artifact import fetch_artifact
from .dataclass import Field
from .formats import Format, SystemFormat
from .inference import InferenceEngine, LogProbInferenceEngine, OpenAiInferenceEngine
from .metrics import BulkInstanceMetric
from .operator import SequentialOperator
from .settings_utils import get_settings
from .system_prompts import EmptySystemPrompt, SystemPrompt
from .templates import Template

settings = get_settings()


def get_task_data_dict(task_data):
    import json

    # seems like the task data sometimes comes as a string, not a dict
    # this fixes it
    return json.loads(task_data) if isinstance(task_data, str) else task_data


class LLMAsJudgeBase(BulkInstanceMetric):
    """LLM-as-judge-based metric class for evaluating correctness.

    Attributes:
        main_score (str): The main score label used for evaluation.
        task (Literal["rating.single_turn"]): The type of task the llm as judge runs. This defines the output and input
         format of the judge model.
        template (Template): The template used when generating inputs for the judge llm.
        format (Format): The format used when generating inputs for judge llm.
        system_prompt (SystemPrompt): The system prompt used when generating inputs for judge llm.
        inference_model (InferenceEngine): The module that creates the inference of the judge llm.
        reduction_map (dict): A dictionary specifying the reduction method for the metric.
        batch_size (int): The size of the bulk.
    """

    main_score: str = "llm_as_judge"
    task: str
    template: Template
    system_prompt: SystemPrompt = Field(default_factory=EmptySystemPrompt)
    format: Format = Field(default_factory=SystemFormat)
    inference_model: InferenceEngine
    reduction_map: Optional[Dict[str, List[str]]] = None
    batch_size: int = 32
    prediction_type = Any  # Because handled with multiple tasks
    infer_log_probs = False
    include_meta_data: bool = False

    def verify(self):
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
            if self.format and type(self.format) is not SystemFormat:
                raise ValueError(
                    "Error in 'LLMAsJudge' metric. Inference model 'OpenAiInferenceEngine' does "
                    "not support formatting. Please remove the format definition from the recipe"
                    " (OpenAi Chat API take care of the formatting automatically)."
                )
            if self.system_prompt and type(self.system_prompt) is not EmptySystemPrompt:
                raise ValueError(
                    "Error in 'LLMAsJudge' metric. Inference model 'OpenAiInferenceEngine' does "
                    "not support system prompt. Please remove the system_prompt definition from the recipe"
                    " (Current implementation of Unitxt does not support this."
                    " Support will be added in future updates)."
                )

    def get_full_task_name(self):
        raise NotImplementedError
        # f"tasks.response_assessment.{self.task}",

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        instances = self.prepare_instances(references, predictions, task_data)
        outputs = infer(
            instances,
            engine=self.inference_model,
            task=self.get_full_task_name(),
            template=self.template,
            system_prompt=self.system_prompt,
            format=self.format,
            return_data=True,
            return_log_probs=self.infer_log_probs,
            return_meta_data=self.include_meta_data,
        )

        return self.get_results_from_outputs(outputs)

    def prepare_instances(self, references, predictions, task_data):
        raise NotImplementedError


class LLMAsJudge(LLMAsJudgeBase):
    """LLM-as-judge-based metric class for evaluating correctness.

    Attributes:
        main_score (str): The main score label used for evaluation.
        task (Literal["rating.single_turn"]): The type of task the llm as judge runs. This defines the output and input
         format of the judge model.
        template (Template): The template used when generating inputs for the judge llm.
        format (Format): The format used when generating inputs for judge llm.
        system_prompt (SystemPrompt): The system prompt used when generating inputs for judge llm.
        strip_system_prompt_and_format_from_inputs (bool): Whether to strip the system prompt and formatting from the
         inputs that the models that is being judges received, when they are inserted to the llm-as-judge prompt.
        inference_model (InferenceEngine): The module that creates the inference of the judge llm.
        reduction_map (dict): A dictionary specifying the reduction method for the metric.
        batch_size (int): The size of the bulk.
    """

    task: Literal[
        "rating.single_turn",
        "rating.single_turn_with_reference",
        "pairwise_comparative_rating.single_turn",
    ]
    strip_system_prompt_and_format_from_inputs: bool = True

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

    def prepare(self):
        super().prepare()
        if self.task == "pairwise_comparative_rating.single_turn":
            self.reduction_map = {"weighted_win_rate": [self.main_score]}
        if self.reduction_map is None:
            self.reduction_map = {"mean": [self.main_score]}

    def verify(self):
        super().verify()
        supported_tasks = [
            "rating.single_turn",
            "rating.single_turn_with_reference",
            "pairwise_comparative_rating.single_turn",
        ]
        assert self.task in supported_tasks, (
            f"Error in 'LLMAsJudge' metric. {self.task} is not a supported task type."
            f"The supported tasks types are: {', '.join(supported_tasks)}."
        )

    def get_full_task_name(self):
        return f"tasks.response_assessment.{self.task}"

    def get_results_from_outputs(self, outputs):
        results = []
        for instance in outputs:
            if self.task == "pairwise_comparative_rating.single_turn":
                task_data = get_task_data_dict(instance["task_data"])
                is_model_b_the_baseline = task_data["model_b"] == "baseline_model"
                if is_model_b_the_baseline:
                    model_a_preference_score = instance["prediction"]
                else:
                    model_a_preference_score = instance["prediction"] * -1

                result = {
                    self.main_score: model_a_preference_score,
                    "judge_raw_output": instance["raw_prediction"],
                    "judge_raw_input": instance["source"],
                }
            else:
                result = {
                    self.main_score: instance["prediction"],
                    "judge_raw_output": instance["raw_prediction"],
                    "judge_raw_input": instance["source"],
                }
            results.append(result)
        return results

    def prepare_instances(self, references, predictions, task_data):
        input_instances = self._get_input_instances(task_data)
        return self._get_instance_for_judge_model(
            input_instances, predictions, references
        )


class TaskBasedLLMasJudge(LLMAsJudgeBase):
    infer_log_probs: bool = True
    mapping: Dict[str, str] = {}
    prediction_field: Optional[str] = None
    include_meta_data: bool = True

    # Allow for input which is a dictionary of all input fields. In this case, all input fields are
    # treated as the task data, and the predictions and references are taken directly from there
    # by the judge's template
    def preprocess_instance(self, instance):
        if "task_data" not in instance:
            instance["task_data"] = instance.copy()
        if "prediction" not in instance:
            instance["prediction"] = None
        if "references" not in instance:
            instance["references"] = [""]
        return instance

    def verify(self):
        super().verify()
        if self.infer_log_probs and not isinstance(
            self.inference_model, LogProbInferenceEngine
        ):
            raise NotImplementedError(
                f"Error in TaskBasedLLMasJudge: return_log_probs set to True but supplied engine "
                f"{self.inference_model.__class__.__name__} does not support logprobs."
            )
        if self.include_meta_data and not hasattr(
            self.inference_model, "get_return_object"
        ):
            Warning(
                f"Supplied inference engine {self.inference_model.__class__.__name__} does not support "
                "return_meta_data. Setting return_meta_data to False. Metadata scores will not appear "
                "in returned instances scores."
            )
            self.include_meta_data = False

    def prepare(self):
        super().prepare()
        self.reduction_map = {"mean": [self.main_score]}

    def get_full_task_name(self):
        return self.task

    def get_results_from_outputs(self, outputs):
        results = []
        for instance in outputs:
            result = {
                self.main_score: instance["prediction"],
                "judge_raw_output": instance["raw_prediction"],
            }
            result.update(instance["infer_meta_data"])
            results.append(result)
        return results

    def prepare_instances(self, references, predictions, task_data):
        from . import get_from_catalog

        instances = []
        judge_task = get_from_catalog(self.get_full_task_name())
        judge_task_input_fields = judge_task.input_fields

        for input_instance, prediction, _ in zip(task_data, predictions, references):
            input_instance = get_task_data_dict(input_instance)

            instance_task_data = {}
            for judge_task_input_field in judge_task_input_fields:
                orig_task_field_name = self.mapping.get(
                    judge_task_input_field, judge_task_input_field
                )
                new_val = input_instance.get(orig_task_field_name)
                if new_val:
                    instance_task_data[judge_task_input_field] = new_val

            if self.prediction_field and prediction:
                instance_task_data[self.prediction_field] = str(prediction)
            instance_task_data = judge_task.process(instance_task_data)["input_fields"]
            instances.append(instance_task_data)

        return instances
