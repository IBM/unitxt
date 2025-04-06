import re
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional

from .api import infer
from .dataclass import Field
from .formats import ChatAPIFormat, Format, SystemFormat
from .inference import InferenceEngine, LogProbInferenceEngine, OpenAiInferenceEngine
from .metrics import BulkInstanceMetric
from .operator import SequentialOperator
from .operators import ArtifactFetcherMixin
from .settings_utils import get_settings
from .system_prompts import EmptySystemPrompt, SystemPrompt
from .templates import Template

settings = get_settings()


def get_task_data_dict(task_data):
    import json

    # seems like the task data sometimes comes as a string, not a dict
    # this fixes it
    return json.loads(task_data) if isinstance(task_data, str) else task_data


class LLMAsJudgeBase(BulkInstanceMetric, ArtifactFetcherMixin):
    """LLM-as-judge-base metric class for evaluating correctness of generated predictions.

    Attributes:
        main_score (str): The main score label used for evaluation.
        task (str): The type of task the llm as judge runs. This defines the output and input
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
    single_reference_per_prediction: bool = True

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
            if self.format and type(self.format) is not ChatAPIFormat:
                if not (
                    type(self.format) is SystemFormat
                    and self.format.__id__ == "formats.empty"
                ):
                    raise ValueError(
                        "Error in 'LLMAsJudge' metric. Inference model 'OpenAiInferenceEngine' does "
                        "not support formatting. Please remove the format definition from the recipe,"
                        "or set the format to either 'formats.empty' or 'formats.chat_api'"
                        " (OpenAi Chat API take care of the formatting automatically)."
                    )
            if self.system_prompt and type(self.system_prompt) is not EmptySystemPrompt:
                raise ValueError(
                    "Error in 'LLMAsJudge' metric. Inference model 'OpenAiInferenceEngine' does "
                    "not support system prompt. Please remove the system_prompt definition from the recipe"
                    " (Current implementation of Unitxt does not support this."
                    " Support will be added in future updates)."
                )

    @abstractmethod
    def get_full_task_name(self):
        pass

    def compute(
        self,
        references: List[List[Any]],
        predictions: List[Any],
        task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        instances = self.prepare_instances(references, predictions, task_data)
        outputs = self.infer_instances(instances)
        return self.get_metric_results_from_prediction_outputs(outputs)

    @abstractmethod
    def prepare_instances(
        self, references, predictions, task_data
    ) -> List[Dict[str, Any]]:
        """Generate a list of instances for inference.

        Each generated instance should include all the fields required by the metrics' task and template, to
        create the source prompt for the judge.
        """
        pass

    @abstractmethod
    def infer_instances(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate the dataset and call the inference engine to generate the judges' predictions.

        Return the list of the produced instances with their generated judge predictions.
        """
        pass

    @abstractmethod
    def get_metric_results_from_prediction_outputs(
        self, outputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate a scores' dictionary for each instance.

        Return the list of scores dictionaries for the input instances.
        """
        pass


class LLMAsJudge(LLMAsJudgeBase):
    """LLM-as-judge-based metric class for evaluating correctness of generated predictions.

    This class uses the source prompt given to the generator and the generator's predictions to evaluate
    correctness using one of three supported tasks (rating.single_turn, rating.single_turn_with_reference,
    pairwise_comparative_rating.single_turn).

    Attributes:
        main_score (str): The main score label used for evaluation.

        task (Literal["rating.single_turn","rating.single_turn_with_reference",
        "pairwise_comparative_rating.single_turn"]): The type of task the llm as judge runs.
        This defines the output and input format of the judge model.

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
                template = self.get_artifact(template)
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
        string_input_instances = []

        for input_instance in input_instances:
            if isinstance(input_instance, str):
                string_input_instances.append(input_instance)
            if isinstance(input_instance, list):  # chat api
                if len(input_instance) == 1:  # only user
                    string_input_instances.append(input_instance[0]["content"])
                if len(input_instance) == 2:  # only system and user
                    string_input_instances.append(
                        input_instance[0]["content"]
                        + "\n"
                        + input_instance[1]["content"]
                    )
                else:  # num demos > 0
                    turns = []
                    for turn in input_instance:
                        turns.append(f"{turn['role']}: {turn['content']}")
                    string_input_instances.append("\n".join(turns))

        if self.task == "rating.single_turn":
            instances = [
                {
                    "question": input_instance,
                    "answer": prediction,
                }
                for input_instance, prediction, reference in zip(
                    string_input_instances, predictions, references
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
                    string_input_instances, predictions, references
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
                    string_input_instances, predictions, references
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

    def infer_instances(self, instances):
        return infer(
            instances,
            engine=self.inference_model,
            task=self.get_full_task_name(),
            template=self.template,
            system_prompt=self.system_prompt,
            format=self.format,
            return_data=True,
        )

    def get_metric_results_from_prediction_outputs(self, outputs):
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
                    f"{self.main_score}_judge_raw_output": instance["raw_prediction"],
                    f"{self.main_score}_judge_raw_input": instance["source"],
                }
            else:
                result = {
                    self.main_score: instance["prediction"],
                    f"{self.main_score}_judge_raw_output": instance["raw_prediction"],
                    f"{self.main_score}_judge_raw_input": instance["source"],
                }
            results.append(result)
        return results

    def prepare_instances(self, references, predictions, task_data):
        input_instances = self._get_input_instances(task_data)
        instances = self._get_instance_for_judge_model(
            input_instances, predictions, references
        )
        # Copy the data classification policy from the original instance
        for instance, single_task_data in zip(instances, task_data):
            instance["data_classification_policy"] = single_task_data.get(
                "metadata", {}
            ).get("data_classification_policy")
        return instances


class TaskBasedLLMasJudge(LLMAsJudgeBase):
    """LLM-as-judge-based metric class for evaluating correctness of generated predictions.

    This class can use any task and matching template to evaluate the predictions. All
    task/templates field are taken from the instance's task_data.
    The instances sent to the judge can either be: 1.a unitxt dataset, in which case the predictions are
    copied to a specified field of the task. 2. dictionaries with the fields required by the task and template.

    Args:
        main_score (str):
            The main score label used for evaluation.
        task (str):
            The type of task the llm as judge runs.
            This defines the output and input format of the judge model.
        template (Template):
            The template used when generating inputs for the judge llm.
        format (Format):
            The format used when generating inputs for judge llm.
        system_prompt (SystemPrompt):
            The system prompt used when generating inputs for judge llm.
        strip_system_prompt_and_format_from_inputs (bool):
            Whether to strip the system prompt and formatting from the
            inputs that the models that is being judges received,
            when they are inserted to the llm-as-judge prompt.
        inference_model (InferenceEngine):
            The module that creates the inference of the judge llm.
        reduction_map (dict):
            A dictionary specifying the reduction method for the metric.
        batch_size (int):
            The size of the bulk.
        infer_log_probs(bool):
            whether to perform the inference using logprobs.
            If true, the template's post-processing must support the logprobs output.
        judge_to_generator_fields_mapping (Dict[str, str]):
            optional mapping between the names of the fields in the generator task and the
            judge task. For example, if the generator task uses "reference_answers" and the judge task  expect "ground_truth",
            include  {"ground_truth": "reference_answers"} in this dictionary.
        prediction_field (str):
            if indicated, and prediction exist, copy prediction to this field name in task_data.
        include_meta_data (bool):
            whether to include the inference per-instance metadata in the returned results.

    """

    infer_log_probs: bool = False
    judge_to_generator_fields_mapping: Dict[str, str] = {}
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
                f"Error in TaskBasedLLMAsJudge: return_log_probs set to True but supplied engine "
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
        self.score_prefix = f"{self.inference_model.get_engine_id()}_"
        if not self.format:
            self.set_format_for_inference_engine()

    # if format is not directly set in constructor, choose according to the inference model
    def set_format_for_inference_engine(self):
        model_name = self.inference_model.get_engine_id()
        if "_wml" in model_name:
            if re.search("llama.?3.*instruct", model_name):
                format_name = "formats.llama3_instruct"
            elif re.search("mixtral", model_name):
                format_name = "formats.models.mistral.instruction"
            else:
                format_name = "formats.empty"
        else:
            format_name = "formats.chat_api"
        self.format = self.get_artifact(format_name)

    def get_full_task_name(self):
        return self.task

    def get_metric_results_from_prediction_outputs(self, outputs):
        results = []
        for instance in outputs:
            result = {
                self.main_score: instance["prediction"],
                f"{self.main_score}_judge_raw_output": instance["raw_prediction"],
                f"{self.main_score}_judge_raw_input": instance["source"],
            }
            if self.include_meta_data:
                meta_data = {
                    f"{self.main_score}_{k}": v
                    for k, v in instance["infer_meta_data"].items()
                }
                result.update(meta_data)
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
                orig_task_field_name = self.judge_to_generator_fields_mapping.get(
                    judge_task_input_field, judge_task_input_field
                )
                new_val = input_instance.get(orig_task_field_name)
                if new_val is None and isinstance(prediction, dict):
                    new_val = prediction.get(orig_task_field_name)
                if new_val is not None:
                    instance_task_data[judge_task_input_field] = new_val

            if self.prediction_field and prediction is not None:
                if isinstance(prediction, dict):
                    prediction = prediction[self.prediction_field]
                instance_task_data[self.prediction_field] = prediction
            instance_task_data = judge_task.process(instance_task_data)["input_fields"]

            data_classification_policy = input_instance.get("metadata", {}).get(
                "data_classification_policy"
            )
            instance_task_data[
                "data_classification_policy"
            ] = data_classification_policy
            instances.append(instance_task_data)

        return instances

    def infer_instances(self, instances):
        return infer(
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
