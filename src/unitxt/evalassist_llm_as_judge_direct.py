import json
from typing import Optional, Union
from unitxt.artifact import fetch_artifact
from unitxt.eval_assist_constants import CriteriaOption, CriteriaWithOptions, EvaluatorNameEnum, ModelFamilyEnum, OptionSelectionStrategyEnum
from .metrics import BulkInstanceMetric
from .inference import InferenceEngine, OptionSelectingByLogProbsInferenceEngine
from .templates import Template
from .task import Task
from .api import infer, select
from unitxt import get_logger

import re
import random

class EvalAssistLLMAsJudgeDirect(BulkInstanceMetric):
    inference_engine: InferenceEngine
    criteria_or_criterias: Optional[Union[CriteriaWithOptions, list[CriteriaWithOptions]]] = None
    assessment_template : Template = None
    summarization_template : Template = None
    option_selection_template : Template = None
    option_selection_strategy: OptionSelectionStrategyEnum = OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT
    evaluator_name: EvaluatorNameEnum = EvaluatorNameEnum.MIXTRAL
    model_family: ModelFamilyEnum = ModelFamilyEnum.MIXTRAL
    check_positional_bias = True
    reduction_map = {"mean": ["score"]}
    main_score = "score"
    logger = get_logger()

    assessment_task = Task(
            input_fields={"context_variables": str, "response": str, "criteria_description": str, "display_options_instruction": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
    
    assessment_task_prometheus = Task(
            input_fields={"score_instructions" : str, "context_variables": str,
                        "response": str, "criteria_description": str, "score_option_instruction" : str},
            reference_fields={},
            prediction_type=str,
            metrics=[])

    summarization_task = Task(
            input_fields={"assessment": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
    
    option_selection_task = Task(
            input_fields={
                "assessment_prompt": str,
                "assessment": str,
                "criteria_description": str,
                "score_option_instruction": str,
                "options": list},
            reference_fields={},
            prediction_type=str,
            metrics=[])
    
    option_selection_task_prometheus = Task(
            input_fields={
                "assessment_prompt": str,
                "assessment": str,
                "options": list},
            reference_fields={},
            prediction_type=str,
            metrics=[])

    def get_parsed_criteria(self, criteria: CriteriaWithOptions):
        criteria_description = criteria.description
        criteria_option_names = [o.name for o in criteria.options]
        
        display_options_instruction = "Choose an answer:\n" + "\n".join([f"- \"{o.name}\"{f' if {o.description}' if o.description != '' else ''}" for o in criteria.options])
        score_option_instruction = "".join([f"Score {o.name}: {o.description}\n" for o in criteria.options])
        
        return criteria_description, criteria_option_names, display_options_instruction, score_option_instruction

    def _parse_completion(self, completion: str, options: list[str]) -> tuple[str, str]:
        """ Ensure that the assessments are always a valid option """
        for o in options:
            search_for = rf"\b{o.strip()}\b"
            match_found = re.search(search_for, completion)
            if match_found is not None:
                return match_found[0]
        # failure case - return a arbitrary option label
        return random.choice(options)
    
    def verify(self):
        super().verify()
        if self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OPTION_LOGPROB and \
            not isinstance(self.inference_engine, OptionSelectingByLogProbsInferenceEngine):
            raise ValueError(
                "The option selection strategy was set to 'PARSE_OPTION_LOGPROB'"
                f"which requires the inference engine '{self.inference_engine.get_pretty_print_name()}' "
                "to inherit from OptionSelectingByLogProbsInferenceEngine "
            )

    def compute(
        self, references: list[list[str]], predictions: list[str], task_data: list[dict[str,any]]
    ) -> dict:
        self.logger.info(f'Starting evaluation with evaluator "{self.evaluator_name}"')
        evaluations_count = len(predictions)
        # TODO: find out how to serialize and deserialize enums
        if isinstance(self.option_selection_strategy, str):
            self.option_selection_strategy = OptionSelectionStrategyEnum[self.option_selection_strategy]
        if isinstance(self.evaluator_name, str):
            self.evaluator_name = EvaluatorNameEnum[self.evaluator_name]
        if isinstance(self.model_family, str):
            self.model_family = ModelFamilyEnum[self.model_family]
            
        self.score_prefix = self.evaluator_name.value + '-'

        if self.criteria_or_criterias is None:
           # Get it from the task data
           # TODO: implement verify to check that the criteria was provided
            criteria_dicts = [task_data_instance["criteria"] for task_data_instance in task_data]
            criterias = [CriteriaWithOptions.from_dict(criteria_dict) for criteria_dict in criteria_dicts]
        # criteria is in passes in the constructor
        elif isinstance(self.criteria_or_criterias, CriteriaWithOptions):
            criterias: list[CriteriaWithOptions] = [self.criteria_or_criterias] * evaluations_count
        elif isinstance(self.criteria_or_criterias, str):
            criteria: CriteriaWithOptions = fetch_artifact(self.criteria_or_criterias)[0]
            criteria.options = [CriteriaOption.from_dict(option) for option in criteria.options]
            criterias = [criteria] * evaluations_count
        else:
            criterias = self.criteria_or_criterias
        assessment_task = self.assessment_task if self.evaluator_name != EvaluatorNameEnum.PROMETHEUS else self.assessment_task_prometheus
        summarization_task = self.summarization_task
        option_selection_task = self.option_selection_task  if self.evaluator_name != EvaluatorNameEnum.PROMETHEUS else self.option_selection_task_prometheus
       
        contexts = [td['context'] for td in task_data]

        if self.check_positional_bias:
            criterias += [CriteriaWithOptions(
                name=criteria.name,
                description=criteria.description,
                option_map=criteria.option_map,
                options=list(reversed(criteria.options))) for criteria in criterias]
            contexts += contexts
            predictions += predictions

        parsed_criterias = [self.get_parsed_criteria(criteria) for criteria in criterias]
        criteria_description_list, criteria_option_names_list, display_options_instruction_list, score_option_instruction_list = zip(*parsed_criterias)

        assessment_for_summaries_slice = slice(0, evaluations_count)

        if self.evaluator_name == EvaluatorNameEnum.PROMETHEUS:
            # Assessment Stage
            asessment_instances = [{
                        "score_instructions" : " or ".join(criteria_option_names),
                        "context_variables": context,
                        "response": prediction,
                        "criteria_description": criteria_description,
                        "score_option_instruction" : score_option_instruction,
                        "data_classification_policy": ["public"]}
                    for context, prediction, criteria_description, criteria_option_names, score_option_instruction in zip(
                        contexts, predictions, criteria_description_list, criteria_option_names_list, score_option_instruction_list)]

            assessment_outputs_dataset = infer(
                asessment_instances,
                task=assessment_task,
                engine=self.inference_engine,
                template=self.assessment_template,
                return_data=True)
            assessment_prompts: list[str] = [instance['source'] for instance in assessment_outputs_dataset]
            assessment_outputs: list[str] = [instance['prediction'] for instance in assessment_outputs_dataset]
            
            self.logger.info("The assessment was generated succesfully.")
            # get the text before [RESULT] as the assessment
            assessment_outputs = [output.split("[RESULT]")[0].strip() for output in assessment_outputs]
            summarization_output = assessment_outputs
            self.logger.info("The summary was generated succesfully.")

            selection_instances = [{
                "assessment_prompt": assessment_prompt, 
                "assessment": assessment_output, 
                "options": criteria_option_names,
            } for assessment_prompt, assessment_output, criteria_option_names in zip(assessment_prompts, assessment_outputs, criteria_option_names_list)]

        else:
            asessment_instances = [{
                        "context_variables": context,
                        "response": prediction,
                        "criteria_description": criteria_description,
                        "display_options_instruction": display_options_instruction,
                        "data_classification_policy": ["public"]
                    }
                    for context, prediction, criteria_description, display_options_instruction in zip(
                        contexts, predictions, criteria_description_list, display_options_instruction_list)]

            assessment_outputs_dataset = infer(
                asessment_instances,
                task=assessment_task,
                engine=self.inference_engine,
                template=self.assessment_template,
                return_data=True)
            assessment_prompts: list[str] = [instance['source'] for instance in assessment_outputs_dataset]
            assessment_outputs: list[str] = [instance['prediction'] for instance in assessment_outputs_dataset]
            
            self.logger.info("The assessment was generated succesfully.")

            # Summarisation Stage
            summarization_instances = [{
                "assessment": assessment_output_instance['prediction'],
                "data_classification_policy": ["public"]
            } for assessment_output_instance in assessment_outputs_dataset[assessment_for_summaries_slice]]

            summarization_output = infer(
                summarization_instances,
                task=summarization_task,
                engine=self.inference_engine,
                template=self.summarization_template)
            
            self.logger.info("The summary was generated succesfully.")

            selection_instances = [{
                "assessment_prompt": assessment_prompt,
                "assessment": assessment_output,
                "criteria_description": criteria_description, 
                "score_option_instruction": score_option_instruction,
                "options": criteria_option_names,
                "data_classification_policy": ["public"]
            } for assessment_prompt, assessment_output, criteria_description, score_option_instruction, criteria_option_names \
                in zip(assessment_prompts, assessment_outputs, criteria_description_list, score_option_instruction_list, criteria_option_names_list)]


        if self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OPTION_LOGPROB:
            option_selection_outputs_dataset = select(
                selection_instances,
                engine=self.inference_engine,
                task=option_selection_task,
                template=self.option_selection_template,
                return_data=True)
            option_selection_prompts: list[str] = [instance['source'] for instance in option_selection_outputs_dataset]
            option_selection_outputs: list[str] = [instance['prediction'] for instance in option_selection_outputs_dataset]
            selections = option_selection_outputs
        elif self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT:
            option_selection_outputs_dataset = infer(
                selection_instances,
                task=option_selection_task,
                engine=self.inference_engine,
                template=self.option_selection_template,
                return_data=True)
            option_selection_prompts: list[str] = [instance['source'] for instance in option_selection_outputs_dataset]
            option_selection_outputs: list[str] = [instance['prediction'] for instance in option_selection_outputs_dataset]
            selections = [self._parse_completion(text_completion, criteria_option_names) for text_completion, criteria_option_names in zip(option_selection_outputs, criteria_option_names_list)]
            
        
        self.logger.info("The selections were calculated succesfully.")


        positional_bias = None
        if self.check_positional_bias:
            positional_bias = [selections[i] != selections[evaluations_count + i] for i in range(evaluations_count)]

        scores = [criteria.option_map[selection] if criteria.option_map is not None else 1 for criteria, selection in zip(criterias, selections)]
        # remove None values from the result dict, e.g. when positional_bias_check is False there is no positional_bias entry in the dict
        return [
            {
                key: value
                for key, value in {
                    "score": scores[i],
                    "positional_bias": positional_bias[i] if self.check_positional_bias else None,
                    "selected_option": selections[i],
                    "positional_bias_selected_option": selections[evaluations_count + i] if self.check_positional_bias else None,
                    "assessment": assessment_outputs_dataset[i]['prediction'],
                    "positional_bias_assessment": assessment_outputs_dataset[i + evaluations_count]['prediction'] if self.check_positional_bias else None,
                    # "option_selection_prompt": option_selection_prompts[i],
                    "summary": summarization_output[i],
                    # "assessment_prompt": assessment_prompts[i],
                    # "positional_bias_assessment_prompt": assessment_prompts[evaluations_count + i],
                    "option_selection_completion": option_selection_outputs[i] if self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT else None,
                    "positional_bias_option_selection_completion": option_selection_outputs[evaluations_count + i] if self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT else None,
                    "option_selection_strategy": self.option_selection_strategy.name,
                }.items()
                if value is not None
            }
            for i in range(evaluations_count)]
            
        