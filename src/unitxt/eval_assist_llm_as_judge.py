from typing import Optional, Union

from .api import infer
from .error_utils import UnitxtError
from .eval_assist_chat_templates import direct_assessment_template_dict
from .eval_assist_constants import (
    Criteria,
    EvaluatorNameEnum,
    OptionSelectionStrategyEnum,
)
from .eval_assist_utils import get_parsed_context
from .inference import (
    InferenceEngine,
    OptionSelectingByLogProbsInferenceEngine,
)
from .logging_utils import get_logger
from .metrics import BulkInstanceMetric
from .task import Task
from .templates import Template


class EvalAssistLLMAsJudge(BulkInstanceMetric):
    inference_engine: InferenceEngine
    option_selection_strategy: OptionSelectionStrategyEnum = (
        OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT
    )
    evaluator_name: EvaluatorNameEnum = None
    check_positional_bias: bool = True
    context_fields: str = ["context"]
    generate_summaries: bool = True
    format = "formats.chat_api"
    include_prompts_in_result: bool = False
    criteria_field: str = None
    criteria: Criteria = None
    logger = get_logger()

    def prepare(self):
        super().prepare()
        if isinstance(self.context_fields, str):
            self.context_fields = [self.context_fields]

        if not isinstance(self.option_selection_strategy, OptionSelectionStrategyEnum):
            self.option_selection_strategy = OptionSelectionStrategyEnum[
                self.option_selection_strategy
            ]
        if not isinstance(self.evaluator_name, EvaluatorNameEnum):
            self.evaluator_name = EvaluatorNameEnum[self.evaluator_name]

        self.assessment_template = direct_assessment_template_dict["assessment"]
        self.summarization_template = direct_assessment_template_dict["summarization"]
        self.option_selection_template = direct_assessment_template_dict["answer"]

        self.assessment_task = Task(
            input_fields={
                "context_variables": str,
                "response": str,
                "criteria_description": str,
                "display_options_instruction": str,
            },
            reference_fields={},
            prediction_type=str,
            metrics=[],
        )

        self.summarization_task = Task(
            input_fields={"assessment": str},
            reference_fields={},
            prediction_type=str,
            metrics=[],
        )

        self.option_selection_task = Task(
            input_fields={
                "context_variables": str,
                "response": str,
                "display_options_instruction": str,
                "assessment": str,
                "criteria_description": str,
                "score_option_instruction": str,
                "options": list,
            },
            reference_fields={},
            prediction_type=str,
            metrics=[],
        )

    def verify(self):
        super().verify()
        if (
            self.option_selection_strategy
            == OptionSelectionStrategyEnum.PARSE_OPTION_LOGPROB
            and not isinstance(
                self.inference_engine, OptionSelectingByLogProbsInferenceEngine
            )
        ):
            raise ValueError(
                "The option selection strategy was set to 'PARSE_OPTION_LOGPROB' "
                f"which requires the inference engine '{self.inference_engine.get_pretty_print_name()}' "
                "to inherit from OptionSelectingByLogProbsInferenceEngine "
            )
        if self.criteria is None and self.criteria_field is None:
            raise UnitxtError(
                f"You must set either the 'criteria' field of the {__class__.__name__} metric to define one criteria to evaluate on all instance, or set a 'criteria_field' of the metric to evaluate on each instance based on the criteria specified in that field of each instance."
            )

    def get_contexts(self, task_data: list[dict[str, any]]) -> list[dict[str, str]]:
        return [
            get_parsed_context(
                {
                    context_field: td[context_field]
                    for context_field in self.context_fields
                }
            )
            for td in task_data
        ]

    def perform_evaluation_step(
        self,
        instances: list,
        task: Task,
        template: Template,
        previous_messages: Optional[list[dict[str, str]]] = None,
    ):
        outputs_dataset = infer(
            instances,
            task=task,
            engine=self.inference_engine,
            template=template,
            format=self.format,
            return_data=True,
            previous_messages=previous_messages,
        )
        prompts: list[str] = [instance["source"] for instance in outputs_dataset]
        raw_predictions: list[str] = [
            instance["raw_prediction"] for instance in outputs_dataset
        ]
        predictions: list[str] = [
            instance["prediction"] for instance in outputs_dataset
        ]
        return (prompts, raw_predictions, predictions)

    def clean_results(self, results: Union[dict, list]):
        if isinstance(results, list):
            return [self.clean_results(x) for x in results]
        cleaned = {
            k: (v if not isinstance(v, dict) else self.clean_results(v))
            for k, v in results.items()
            if v is not None and not (isinstance(v, (list, dict)) and len(v) == 0)
        }
        # Remove the dictionary itself if it becomes empty
        return {
            k: v
            for k, v in cleaned.items()
            if not (isinstance(v, dict) and len(v) == 0)
        }
