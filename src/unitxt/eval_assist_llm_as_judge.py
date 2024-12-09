import random
import re

from . import get_logger
from .api import infer, select
from .artifact import fetch_artifact
from .eval_assist_chat_templates import direct_assessment_template_dict
from .eval_assist_constants import (
    Criteria,
    CriteriaWithOptions,
    EvaluatorNameEnum,
    OptionSelectionStrategyEnum,
)
from .eval_assist_utils import get_parsed_context
from .inference import (
    InferenceEngine,
    NoInputLogProbsExeption,
    OptionSelectingByLogProbsInferenceEngine,
)
from .metrics import BulkInstanceMetric
from .task import Task


class EvalAssistLLMAsJudge(BulkInstanceMetric):
    inference_engine: InferenceEngine
    criteria_or_criterias: Criteria = None
    option_selection_strategy: OptionSelectionStrategyEnum = None
    evaluator_name: EvaluatorNameEnum = None
    check_positional_bias = True
    context_field: str = 'context'

    logger = get_logger()

    def prepare(self):
        super().prepare()
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

    def get_parsed_criteria(self, criteria: CriteriaWithOptions):
        criteria_description = criteria.description
        criteria_option_names = [o.name for o in criteria.options]

        display_options_instruction = "Choose an answer:\n" + "\n".join(
            [
                f"- \"{o.name}\"{f' if {o.description}' if o.description != '' else ''}"
                for o in criteria.options
            ]
        )
        score_option_instruction = "".join(
            [f"Score {o.name}: {o.description}\n" for o in criteria.options]
        )

        return (
            criteria_description,
            criteria_option_names,
            display_options_instruction,
            score_option_instruction,
        )

    def get_contexts(self, task_data: list[dict[str, any]]) -> list[dict[str,str]]:
        contexts: list[dict[str,str] | str] = [td[self.context_field] for td in task_data]
        return [
            get_parsed_context(context if isinstance(context, dict) else {'context': context})
            for context in contexts
        ]