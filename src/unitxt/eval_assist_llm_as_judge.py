import random
import re

from . import get_logger
from .api import infer, select
from .artifact import fetch_artifact
from .eval_assist_constants import (
    Criteria,
    CriteriaWithOptions,
    EvaluatorNameEnum,
    ModelFamilyEnum,
    OptionSelectionStrategyEnum,
)
from .eval_assist_chat_templates import direct_assessment_template_dict
from .eval_assist_utils import get_parsed_context
from .inference import InferenceEngine, NoInputLogProbsExeption, OptionSelectingByLogProbsInferenceEngine
from .metrics import BulkInstanceMetric
from .task import Task
from .templates import Template


class EvalAssistLLMAsJudge(BulkInstanceMetric):
    inference_engine: InferenceEngine
    criteria_or_criterias: Criteria = None
    option_selection_strategy: OptionSelectionStrategyEnum = None
    evaluator_name: EvaluatorNameEnum = None
    model_family: ModelFamilyEnum = None
    check_positional_bias = True
    use_score_prefix = False

    logger = get_logger()


    def prepare(self):
        super().prepare()
        if self.use_score_prefix:
            self.score_prefix = self.evaluator_name.value + "-"
        if isinstance(self.option_selection_strategy, str):
            self.option_selection_strategy = OptionSelectionStrategyEnum[
                self.option_selection_strategy
            ]
        if isinstance(self.evaluator_name, str):
            self.evaluator_name = EvaluatorNameEnum[self.evaluator_name]
        if isinstance(self.model_family, str):
            self.model_family = ModelFamilyEnum[self.model_family]

        self.format = "formats.chat_api"

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

    def _parse_completion(self, completion: str, options: list[str]) -> tuple[str, str]:
        """Ensure that the assessments are always a valid option."""
        for o in options:
            search_for = rf"\b{o.strip()}\b"
            match_found = re.search(search_for, completion)
            if match_found is not None:
                return match_found[0]
        # failure case - return a arbitrary option label
        return random.choice(options)
        

    def compute(
        self,
        references: list[list[str]],
        predictions: list[str],
        task_data: list[dict[str, any]],
    ) -> dict:
        self.logger.info(
            f'Starting evaluation with evaluator "{self.evaluator_name}" and provider {self.inference_engine.get_pretty_print_name()}'
        )
        evaluations_count = len(predictions)
        # TODO: find out how to serialize and deserialize enums
        
        if self.criteria_or_criterias is None:
            self.logger.info("Reading criteria from the task_data")
            criteria_dicts = [
                {**task_data_instance["criteria"], "__type__": "criteria_with_options"}
                for task_data_instance in task_data
            ]
            for criteria_dict in criteria_dicts:
                criteria_dict["options"] = [
                    {**option, "__type__": "criteria_option"}
                    for option in criteria_dict["options"]
                ]
            criterias = [
                fetch_artifact(criteria_dict)[0] for criteria_dict in criteria_dicts
            ]
        # criteria is in passes in the constructor
        elif isinstance(self.criteria_or_criterias, CriteriaWithOptions):
            self.logger.info(
                "Reading criteria from self. Criteria is a single CriteriaWithOptions, replicating it for all predictions"
            )
            criterias: list[CriteriaWithOptions] = [
                self.criteria_or_criterias
            ] * evaluations_count
        else:
            criterias = self.criteria_or_criterias

        contexts = [
            get_parsed_context(context)
            for context in [td["context"] for td in task_data]
        ]
        if self.check_positional_bias:
            criterias += [
                CriteriaWithOptions(
                    name=criteria.name,
                    description=criteria.description,
                    option_map=criteria.option_map,
                    options=list(reversed(criteria.options)),
                )
                for criteria in criterias
            ]
            contexts += contexts
            predictions += predictions

        parsed_criterias = [self.get_parsed_criteria(criteria) for criteria in criterias]
        
        (
            criteria_description_list,
            criteria_option_names_list,
            display_options_instruction_list,
            score_option_instruction_list,
        ) = zip(*parsed_criterias)

        assessment_for_summaries_slice = slice(0, evaluations_count)

        assessment_instances = [
            {
                "context_variables": context,
                "response": prediction,
                "display_options_instruction": display_options_instruction,
                "criteria_description": criteria_description,
                "data_classification_policy": ["public"],
            }
            for context, prediction, criteria_description, display_options_instruction in zip(
                contexts,
                predictions,
                criteria_description_list,
                display_options_instruction_list,
            )
        ]

        assessment_outputs_dataset = infer(
            assessment_instances,
            task=self.assessment_task,
            engine=self.inference_engine,
            template=self.assessment_template,
            format=self.format,
            return_data=True,
        )
        assessment_prompts: list[str] = [
            instance["source"] for instance in assessment_outputs_dataset
        ]

        assessment_outputs: list[str] = [
            instance["prediction"] for instance in assessment_outputs_dataset
        ]

        self.logger.info("The assessment was generated successfully.")
        # Summarisation Stage
        summarization_instances = [
            {
                "assessment": assessment_output,
                "data_classification_policy": ["public"],
            }
            for assessment_output in assessment_outputs[
                assessment_for_summaries_slice
            ]
        ]

        summarization_output = infer(
            summarization_instances,
            task=self.summarization_task,
            engine=self.inference_engine,
            template=self.summarization_template,
            format=self.format,
        )

        self.logger.info("The summary was generated successfully.")

        selection_instances = [
            {
                "context_variables": context,
                "response": prediction,
                "display_options_instruction": display_options_instruction,
                "assessment": assessment_output,
                "criteria_description": criteria_description,
                "score_option_instruction": score_option_instruction,
                "options": criteria_option_names,
                "data_classification_policy": ["public"],
            }
            for assessment_output, criteria_description, score_option_instruction, criteria_option_names, context, prediction, display_options_instruction in zip(
                assessment_outputs,
                criteria_description_list,
                score_option_instruction_list,
                criteria_option_names_list,
                contexts,
                predictions,
                display_options_instruction_list,
            )
        ]
        parse_output_logprobs_failed = False
        if self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OPTION_LOGPROB:
            try:
                option_selection_outputs_dataset = select(
                    selection_instances,
                    engine=self.inference_engine,
                    task=self.option_selection_task,
                    template=self.option_selection_template,
                    format=self.format,
                    return_data=True,
                )
                option_selection_prompts: list[str] = [
                    instance["source"] for instance in option_selection_outputs_dataset
                ]
                option_selection_outputs: list[str] = [
                    instance["prediction"] for instance in option_selection_outputs_dataset
                ]
                selections = option_selection_outputs
            except NoInputLogProbsExeption as e:
                self.logger.error(f"An error occurred: {e}")
                self.logger.warning(f'{self.option_selection_strategy.name} failed. trying {OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT.name} instead.')
                parse_output_logprobs_failed = True

        if self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT or parse_output_logprobs_failed:
            option_selection_outputs_dataset = infer(
                selection_instances,
                task=self.option_selection_task,
                engine=self.inference_engine,
                template=self.option_selection_template,
                format=self.format,
                return_data=True,
            )
            option_selection_prompts: list[str] = [
                instance["source"] for instance in option_selection_outputs_dataset
            ]
            option_selection_outputs: list[str] = [
                    instance["raw_prediction"] for instance in option_selection_outputs_dataset
                ]
            selections: list[str] = [
                    instance["prediction"] for instance in option_selection_outputs_dataset
                ]
            
            print('option_selection_outputs')
            print(option_selection_outputs)
            print('selections')
            print(selections)
            
        self.logger.info("The selections were calculated successfully.")

        positional_bias = None
        if self.check_positional_bias:
            positional_bias = [
                selections[i] != selections[evaluations_count + i]
                for i in range(evaluations_count)
            ]

        scores = [
            criteria.option_map[selection] if criteria.option_map is not None else 1
            for criteria, selection in zip(criterias, selections)
        ]
        # remove None values from the result dict, e.g. when positional_bias_check is False there is no positional_bias entry in the dict
        return [
            {
                key: value
                for key, value in {
                    "score": scores[i],
                    "positional_bias": positional_bias[i] if self.check_positional_bias else None,
                    "selected_option": selections[i],
                    "positional_bias_selected_option": selections[evaluations_count + i] if self.check_positional_bias else None,
                    "assessment": assessment_outputs_dataset[i]["prediction"],
                    "positional_bias_assessment": assessment_outputs_dataset[i + evaluations_count]["prediction"] if self.check_positional_bias else None,
                    "option_selection_prompt": option_selection_prompts[i],
                    "posional_bias_option_selection_prompt": option_selection_prompts[i + evaluations_count],
                    "summary": summarization_output[i],
                    # "assessment_prompt": assessment_prompts[i],
                    # "positional_bias_assessment_prompt": assessment_prompts[evaluations_count + i],
                    "option_selection_completion": option_selection_outputs[i] if self.option_selection_strategy== OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT else None,
                    "positional_bias_option_selection_completion": option_selection_outputs[evaluations_count + i] if self.option_selection_strategy== OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT else None,
                    "option_selection_strategy": self.option_selection_strategy.name,
                }.items()
                if value is not None
            }
            for i in range(evaluations_count)
        ]
