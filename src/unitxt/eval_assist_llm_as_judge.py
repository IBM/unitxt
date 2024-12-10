from . import get_logger
from .eval_assist_chat_templates import direct_assessment_template_dict
from .eval_assist_constants import (
    Criteria,
    EvaluatorNameEnum,
    OptionSelectionStrategyEnum,
)
from .eval_assist_utils import get_parsed_context
from .inference import (
    InferenceEngine,
)
from .metrics import BulkInstanceMetric
from .task import Task


class EvalAssistLLMAsJudge(BulkInstanceMetric):
    inference_engine: InferenceEngine
    criteria: Criteria = None
    option_selection_strategy: OptionSelectionStrategyEnum = None
    evaluator_name: EvaluatorNameEnum = None
    check_positional_bias = True
    context_fields: str = ["context"]
    generate_summaries: bool = True
    logger = get_logger()

    def prepare(self):
        super().prepare()
        if isinstance(self.context_fields, str):
            self.context_fields = [self.context_fields]

        self.format = "formats.chat_api"
        if isinstance(self.option_selection_strategy, str):
            self.option_selection_strategy = OptionSelectionStrategyEnum[
                self.option_selection_strategy
            ]
        if isinstance(self.evaluator_name, str):
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
