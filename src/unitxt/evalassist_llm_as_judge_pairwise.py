import itertools
import random
import re
from unitxt.artifact import fetch_artifact
from unitxt.eval_assist_constants import Criteria, EvaluatorNameEnum, ModelFamilyEnum, OptionSelectionStrategyEnum
from unitxt.formats import OpenAIFormat, SystemFormat
from .metrics import BulkInstanceMetric
from .inference import InferenceEngine
from unitxt import get_logger
from .templates import Template
from .task import Task
from .api import infer, select
from typing import Optional, Union


def rank_indexes(numbers):
    # Generate the initial list of indices
    indices = list(range(len(numbers)))
    
    # Sort the indices based on the corresponding values in numbers (descending order)
    sorted_indices = sorted(indices, key=lambda x: -numbers[x])
    
    # Initialize a list to hold the rankings
    rankings = [0] * len(numbers)
    
    # Assign rankings
    current_rank = 0
    for i in range(len(sorted_indices)):
        if i > 0 and numbers[sorted_indices[i]] != numbers[sorted_indices[i - 1]]:
            current_rank = i
        rankings[sorted_indices[i]] = current_rank
    
    return rankings

class EvalAssistLLMAsJudgePairwise(BulkInstanceMetric):
    inference_engine: InferenceEngine
    criteria_or_criterias: Optional[Union[str, Criteria, list[Criteria]]] = None
    assessment_template : Template = None
    summarization_template : Template = None
    option_selection_template : Template = None
    check_positional_bias = True
    option_selection_strategy: OptionSelectionStrategyEnum = OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT
    evaluator_name: EvaluatorNameEnum = EvaluatorNameEnum.MIXTRAL
    model_family: ModelFamilyEnum = ModelFamilyEnum.MIXTRAL
    reduction_map = {"mean": ["winrate"]}
    main_score = "winrate"
    logger = get_logger()

    assessment_task =  Task(
            input_fields={"context_variables": str, "response_a" : str, "response_b" : str,
                "option_a" : str, "option_b" : str, "criteria_name" : str, "criteria_description" : str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
    
    assessment_task_prometheus =  Task(
            input_fields={"context_variables": str, "response_a" : str, "response_b" : str,
                "option_a" : str, "option_b" : str, "criteria": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
    
    summarization_task =  Task(
            input_fields={"assessment": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
    
    option_selection_task = Task(
            input_fields={
                "assessment_prompt": str,
                "assessment": str,
                "choose_response_instruction": str,
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

    def _parse_completion(self, completion: str, options: list[str]) -> tuple[str, str]:
        """ Ensure that the assessments are always a valid option """
        for o in options:
            search_for = rf"\b{o.strip()}\b"
            match_found = re.search(search_for, completion)
            if match_found is not None:
                return match_found[0]
        # failure case - return a arbitrary option label
        return random.choice(options)
    
    def compute(
        self, references: list[list[str]], predictions: list[str], task_data: list[dict[str,str]]
    ) -> dict:
        self.logger.info(f'Starting evaluation with evaluator "{self.evaluator_name}"')
        evaluations_count = len(predictions)
        combination_indexes = list(itertools.combinations(range(evaluations_count), 2))
        contests_count = len(combination_indexes)
        per_response_results = {f"{i+1}": {'summaries': [], 'contest_results': [], 'selections': [], 'compared_to': [], "completions": [], "positional_bias_completions": []} for i in range(evaluations_count)}
        response_pairs: list[list[str]] = []
        option_pairs: list[list[str]] = []
        for combination in combination_indexes:
            (response_name_1, response_name_2) = combination
            response_pairs.append([predictions[response_name_1], predictions[response_name_2]])
            option_pairs.append([f'{response_name_1 + 1}', f'{response_name_2 + 1}'])
    
        # TODO: find out how to serialize and deserialize enums
        if isinstance(self.option_selection_strategy, str):
            self.option_selection_strategy = OptionSelectionStrategyEnum[self.option_selection_strategy]
        if isinstance(self.evaluator_name, str):
            self.evaluator_name = EvaluatorNameEnum[self.evaluator_name]
        if isinstance(self.model_family, str):
            self.model_family = ModelFamilyEnum[self.model_family]

        self.score_prefix = self.evaluator_name.value + '-'

        format = OpenAIFormat() if self.model_family == ModelFamilyEnum.GPT else SystemFormat()

        if self.criteria_or_criterias is None:
            #Get it from the task data
            # TODO: implement verify to check that the criteria was provided
            criteria_dicts = [task_data_instance["criteria"] for task_data_instance in task_data]
            criterias = [Criteria.from_dict(criteria_dict) for criteria_dict in criteria_dicts]
        # criteria is in passes in the constructor
        elif isinstance(self.criteria_or_criterias, Criteria):
            criterias: list[Criteria] = [self.criteria_or_criterias] * contests_count
        elif isinstance(self.criteria_or_criterias, str):
            criteria: Criteria = fetch_artifact(self.criteria_or_criterias)[0]
            criterias = [criteria] * contests_count
        else:
            criterias = self.criteria_or_criterias
        assessment_task = self.assessment_task if self.evaluator_name != EvaluatorNameEnum.PROMETHEUS else self.assessment_task_prometheus
        summarization_task = self.summarization_task
        option_selection_task = self.option_selection_task  if self.evaluator_name != EvaluatorNameEnum.PROMETHEUS else self.option_selection_task_prometheus
       
        contexts = [td['context'] for td in task_data]

        if self.check_positional_bias:
            criterias += criterias
            contexts += contexts
            predictions += predictions
            response_pairs += [list(reversed(response_pair)) for response_pair in response_pairs]
            option_pairs += [list(reversed(option_pair)) for option_pair in option_pairs]

        assessment_for_summaries_slice = slice(0, contests_count)

        if self.evaluator_name == EvaluatorNameEnum.PROMETHEUS:
            # Assessment Stage
            asessment_instances = [{
                        "context_variables": context,
                        "response_a": response_pair[0],
                        "response_b": response_pair[1],
                        "option_a": option_pair[0],
                        "option_b": option_pair[1],
                        "criteria": f"{criteria.name}: {criteria.description}",
                        "data_classification_policy": ["public"]}
                    for context, response_pair, option_pair, criteria in zip(
                        contexts, response_pairs, option_pairs, criterias)]

            assessment_outputs_dataset = infer(
                asessment_instances,
                task=assessment_task,
                engine=self.inference_engine,
                template=self.assessment_template,
                return_data=True,
                format=format)
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
                "options": [f"Response {option}" for option in option_pair],
            } for assessment_prompt, assessment_output, option_pair \
                in zip(assessment_prompts, assessment_outputs, option_pairs)]

        else:
            asessment_instances = [{
                        "context_variables": context,
                        "response_a": response_pair[0],
                        "response_b": response_pair[1],
                        "option_a": option_pair[0],
                        "option_b": option_pair[1],
                        "criteria_name": criteria.name,
                        "criteria_description": criteria.description,
                        "data_classification_policy": ["public"]
                    }
                    for context, criteria, response_pair, option_pair in zip(
                        contexts, criterias, response_pairs, option_pairs)]

            assessment_outputs_dataset = infer(
                asessment_instances,
                task=assessment_task,
                engine=self.inference_engine,
                template=self.assessment_template,
                return_data=True,
                format=format)
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
                template=self.summarization_template,
                format=format)
            
            self.logger.info("The summary was generated succesfully.")

            choose_response_instructions = ["".join([f"Choose \"{option}\" if Response {option} is better quality.\n" for option in option_pair]) for option_pair in option_pairs]
            asessment_instances = [{
                        "context_variables": context,
                        "response": prediction,
                        "choose_response_instruction": choose_response_instruction,
                        "data_classification_policy": ["public"]
                    }
                    for context, prediction, choose_response_instruction in zip(
                        contexts, predictions, choose_response_instructions)]
            
            selection_instances = [{
                "assessment_prompt": assessment_prompt,
                "assessment": assessment_output,
                "choose_response_instruction": choose_response_instruction,
                "options": [f"Response {option}" for option in option_pair],
                "data_classification_policy": ["public"]
            } for assessment_prompt, assessment_output, choose_response_instruction, option_pair \
                in zip(assessment_prompts, assessment_outputs, choose_response_instructions, option_pairs)]

        if self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OPTION_LOGPROB:
            option_selection_outputs_dataset = select(
                selection_instances,
                engine=self.inference_engine,
                task=option_selection_task,
                template=self.option_selection_template,
                return_data=True,
                format=format)
            option_selection_prompts: list[str] = [instance['source'] for instance in option_selection_outputs_dataset]
            option_selection_outputs: list[str] = [instance['prediction'] for instance in option_selection_outputs_dataset]
            # take the number of the response from 'Response n'
            selections = [selection.split(' ')[-1] for selection in option_selection_outputs]
        elif self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT:
            option_selection_outputs_dataset = infer(
                selection_instances,
                task=option_selection_task,
                engine=self.inference_engine,
                template=self.option_selection_template,
                return_data=True,
                format=format)
            option_selection_prompts: list[str] = [instance['source'] for instance in option_selection_outputs_dataset]
            option_selection_outputs: list[str] = [instance['prediction'] for instance in option_selection_outputs_dataset]
            selections = [self._parse_completion(text_completion, option_pair) for text_completion, option_pair in zip(option_selection_outputs, option_pairs)]
     
        self.logger.info("The selections were calculated succesfully.")

        ### process results

        positional_bias = None
        if self.check_positional_bias:
            positional_bias = [selections[i] != selections[contests_count + i] for i in range(contests_count)]
        for i in range(contests_count):
            response_key = f"{i+1}"
            (idx_1, idx_2) = combination_indexes[i]
            response_name_1 = f"{idx_1+1}"
            response_name_2 = f"{idx_2+1}"
            # add contest results
            selected_response_name = selections[i]
            per_response_results[response_name_1]['contest_results'].append(selected_response_name == response_name_1)
            per_response_results[response_name_2]['contest_results'].append(selected_response_name == response_name_2)
            per_response_results[response_name_1]['selections'].append(selected_response_name)
            per_response_results[response_name_2]['selections'].append(selected_response_name)
            
            # add the response indexes to which the response was compared to
            per_response_results[response_name_1]['compared_to'].append(f"{response_name_2}")
            per_response_results[response_name_2]['compared_to'].append(f"{response_name_1}")

            # add summaries
            per_response_results[response_name_1]['summaries'].append(summarization_output[i])
            per_response_results[response_name_2]['summaries'].append(summarization_output[i])

            ## add positional bias
            if self.check_positional_bias:
                if not 'positional_bias' in per_response_results[response_name_1]:
                    per_response_results[response_name_1]['positional_bias'] = []
                if not 'positional_bias' in per_response_results[response_name_2]:
                    per_response_results[response_name_2]['positional_bias'] = []
                per_response_results[response_name_1]['positional_bias'].append(positional_bias[i])
                per_response_results[response_name_2]['positional_bias'].append(positional_bias[i])

            if self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT:
                per_response_results[response_name_1]['completions'].append(option_selection_outputs[i])
                per_response_results[response_name_2]['completions'].append(option_selection_outputs[i])
                if self.check_positional_bias:
                    per_response_results[response_name_1]['positional_bias_completions'].append(option_selection_outputs[i])
                    per_response_results[response_name_2]['positional_bias_completions'].append(option_selection_outputs[i])

        # add winrate
        for key in per_response_results:
            contest_results = per_response_results[key]['contest_results']
            winrate = sum(contest_results) / len(contest_results)
            per_response_results[key]['winrate'] = winrate
        # calculate ranking
        ranking = rank_indexes([result['winrate'] for result in per_response_results.values()])
        for i in range(len(ranking)):
            per_response_results[response_key]['ranking'] = ranking[i] + 1

        for key in per_response_results:
            per_response_results[key]['response_name'] = key
            
        # remove None values from the result dict, e.g. when positional_bias_check is False there is no positional_bias entry in the dict
        return [
            {
                key: value
                for key, value in per_response_results[key].items()
                if value is not None or (isinstance(value, list) and len(value) > 0)
            }
            for key in [f"{i+1}" for i in range(evaluations_count)]]
        