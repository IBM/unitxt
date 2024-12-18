import itertools
from difflib import get_close_matches
from typing import List

from .artifact import fetch_artifact
from .error_utils import UnitxtError
from .eval_assist_chat_templates import pairwise_comparison_template_dict
from .eval_assist_constants import (
    Criteria,
    OptionSelectionStrategyEnum,
)
from .eval_assist_llm_as_judge import EvalAssistLLMAsJudge
from .task import Task


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


class EvalAssistLLMAsJudgePairwise(EvalAssistLLMAsJudge):
    criteria: Criteria = None
    criteria_field: str = None
    reduction_map = {"mean": ["score"]}
    main_score = "score"
    prediction_type = List[str]

    def prepare(self):
        super().prepare()
        self.assessment_template = pairwise_comparison_template_dict["assessment"]
        self.summarization_template = pairwise_comparison_template_dict["summarization"]
        self.option_selection_template = pairwise_comparison_template_dict["answer"]

        self.assessment_task = Task(
            input_fields={
                "context_variables": str,
                "response_a": str,
                "response_b": str,
                "option_a": str,
                "option_b": str,
                "criteria_name": str,
                "criteria_description": str,
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
                "score_option_instruction": str,
                "options": list,
            },
            reference_fields={},
            prediction_type=str,
            metrics=[],
        )

    def verify(self):
        if self.criteria is None and self.criteria_field is None:
            raise UnitxtError(
                f"You must set either the 'Criteria' field of the {__class__.__name__} metric to define one criteria to evaluate on all instance, or set a 'criteria_field' of the metric to evaluate on each instance based on the criteria specified in that field of each instance."
            )

    def get_criterias(self, task_data, eval_count):
        if self.criteria is None:
            if self.criteria_field not in task_data[0]:
                raise UnitxtError(
                    f"The criteria field `{self.criteria_field}` required for {__class__.__name__} is not found in instance.  Perhaps you meant '{get_close_matches(self.criteria_field, task_data[0].keys(), n=1, cutoff=0.0)[0]}'?"
                )
            self.logger.info(
                f"Reading criteria from the task_data field f{self.criteria_field}"
            )
            criterias = [
                fetch_artifact(task_data_instance[self.criteria_field])[0]
                for task_data_instance in task_data
            ]
        else:
            self.logger.info(
                "Reading criteria from self. Criteria is a single Criteria, replicating it for all predictions"
            )
            if not isinstance(self.criteria, Criteria):
                raise UnitxtError(
                    f"The type of the criteria must be 'Criteria', instead it is of type '{type(self.criteria)}'"
                )

            criterias: list[Criteria] = [self.criteria] * eval_count

        unique_criterias = list({criteria.name for criteria in criterias})
        self.logger.info(f"Criteria names are '{', '.join(unique_criterias)}'")        
        return criterias

    def get_instance_results(
        self,
        instance_predictions: dict[str, str],
        assessment_prompts,
        assessment_outputs,
        summarization_prompts,
        summarization_outputs,
        option_selection_prompts,
        option_selection_outputs,
        selections,
        contests_count,
        combination_indexes,
    ):
        response_names = list(instance_predictions.keys())
        per_response_results = {
            response_key: {
                "summaries": [],
                "contest_results": [],
                "selections": [],
                "compared_to": [],
                "assessments": [],
                "positional_bias_assessments": [],
                "option_selection_outputs": [],
                "positional_bias": [],
                "positional_bias_selection": [],
                "prompts": {
                    "assessment": [],
                    "positional_bias_assessment": [],
                    "option_selection": [],
                    "positional_bias_option_selection": [],
                    "summary": [],
                },
            }
            for response_key in response_names
        }

        positional_bias = None
        for i in range(contests_count):
            positional_bias_i = contests_count + i
            (idx_1, idx_2) = combination_indexes[i]
            response_name_1 = response_names[idx_1]
            response_name_2 = response_names[idx_2]
            # add contest results
            selected_response_name = selections[i]
            per_response_results[response_name_1]["contest_results"].append(
                selected_response_name == response_name_1
            )
            per_response_results[response_name_2]["contest_results"].append(
                selected_response_name == response_name_2
            )
            per_response_results[response_name_1]["assessments"].append(
                assessment_outputs[i]
            )
            per_response_results[response_name_2]["assessments"].append(
                assessment_outputs[i]
            )
            per_response_results[response_name_1]["positional_bias_assessments"].append(
                assessment_outputs[positional_bias_i]
            )
            per_response_results[response_name_2]["positional_bias_assessments"].append(
                assessment_outputs[positional_bias_i]
            )
            per_response_results[response_name_1]["selections"].append(
                selected_response_name
            )
            per_response_results[response_name_2]["selections"].append(
                selected_response_name
            )

            # add the response indexes to which the response was compared to
            per_response_results[response_name_1]["compared_to"].append(
                f"{response_name_2}"
            )
            per_response_results[response_name_2]["compared_to"].append(
                f"{response_name_1}"
            )

            if self.include_prompts_in_result:
                per_response_results[response_name_1]["prompts"]["assessment"].append(
                    assessment_prompts[i]
                )
                per_response_results[response_name_2]["prompts"]["assessment"].append(
                    assessment_prompts[i]
                )
            if self.generate_summaries:
                # add summaries
                if self.include_prompts_in_result:
                    per_response_results[response_name_1]["prompts"]["summary"].append(
                        summarization_prompts[i]
                    )
                    per_response_results[response_name_2]["prompts"]["summary"].append(
                        summarization_prompts[i]
                    )
                per_response_results[response_name_1]["summaries"].append(
                    summarization_outputs[i]
                )
                per_response_results[response_name_2]["summaries"].append(
                    summarization_outputs[i]
                )
            if self.include_prompts_in_result:
                per_response_results[response_name_1]["prompts"][
                    "option_selection"
                ].append(option_selection_prompts[i])
                per_response_results[response_name_2]["prompts"][
                    "option_selection"
                ].append(option_selection_prompts[i])

            ## add positional bias
            if self.check_positional_bias:
                positional_bias = selections[i] != selections[positional_bias_i]

                per_response_results[response_name_1]["positional_bias"].append(
                    positional_bias
                )
                per_response_results[response_name_2]["positional_bias"].append(
                    positional_bias
                )

                # add prompts
                if self.include_prompts_in_result:
                    per_response_results[response_name_1]["prompts"][
                        "positional_bias_assessment"
                    ].append(assessment_prompts[positional_bias_i])
                    per_response_results[response_name_2]["prompts"][
                        "positional_bias_assessment"
                    ].append(assessment_prompts[positional_bias_i])
                    per_response_results[response_name_1]["prompts"][
                        "positional_bias_option_selection"
                    ].append(option_selection_prompts[positional_bias_i])
                    per_response_results[response_name_2]["prompts"][
                        "positional_bias_option_selection"
                    ].append(option_selection_prompts[positional_bias_i])

            if (
                self.option_selection_strategy
                == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT
            ):
                per_response_results[response_name_1][
                    "option_selection_outputs"
                ].append(option_selection_outputs[i])
                per_response_results[response_name_2][
                    "option_selection_outputs"
                ].append(option_selection_outputs[i])
                if self.check_positional_bias:
                    per_response_results[response_name_1][
                        "positional_bias_selection"
                    ].append(option_selection_outputs[positional_bias_i])
                    per_response_results[response_name_2][
                        "positional_bias_selection"
                    ].append(option_selection_outputs[positional_bias_i])

        # add winrate
        for key in response_names:
            contest_results = per_response_results[key]["contest_results"]
            winrate = sum(contest_results) / len(contest_results)
            per_response_results[key]["winrate"] = winrate
            per_response_results[key]["llm_as_a_judge_score"] = winrate
        # calculate ranking
        ranking = rank_indexes(
            [result["winrate"] for result in per_response_results.values()]
        )

        for response_name, r_i in zip(response_names, ranking):
            per_response_results[response_name]["ranking"] = ranking[r_i] + 1

        for response_name in response_names:
            # add response name
            per_response_results[response_name]["response_name"] = response_name

        all_results = {}
        for response_name in response_names:
            single_result = per_response_results[response_name]
            for metric in single_result.keys():
                all_results[f"{response_name}_{metric}"] = single_result[metric]

        winrates = [r["winrate"] for r in per_response_results.values()]
        all_results["score"] = max(range(len(winrates)), key=winrates.__getitem__)
        all_results['criteria'] = criteria.to_json()
        return self.clean_results(all_results)

    def parse_prediction_to_dict(self, prediction: dict[str, str] | list[str]):
        if isinstance(prediction, list):
            return {f"{key + 1}": value for key, value in enumerate(prediction)}

        if isinstance(prediction, dict):
            return prediction

        raise Exception(
            f"Prediction may be a list or a dict. Instead got type {type(prediction)}"
        )

    def convert_predictions_to_dicts(
        self, predictions: list[dict[str, str] | list[str]]
    ):
        return [self.parse_prediction_to_dict(prediction) for prediction in predictions]

    def compute(
        self,
        references: list[list[str]],
        predictions: list[dict[str, str] | list[str]],
        task_data: list[dict[str, str]],
    ) -> dict:
        self.logger.info(
            f'Starting evaluation with evaluator "{self.evaluator_name}" and provider {self.inference_engine.get_pretty_print_name()}'
        )
        predictions = self.convert_predictions_to_dicts(predictions)
        instances_count = len(predictions)
        self.reduction_map["mean"].extend(
            [f"{key}_winrate" for key in predictions[0].keys()]
        )
        self.reduction_map["mean"].extend(
            [f"{key}_ranking" for key in predictions[0].keys()]
        )

        predictions_count_list = [len(prediction) for prediction in predictions]
        combination_indexes_list = [
            list(itertools.combinations(range(evaluations_count), 2))
            for evaluations_count in predictions_count_list
        ]
        contests_count_list = [
            len(combination_indexes) for combination_indexes in combination_indexes_list
        ]

        self.logger.info(
            f"The evaluation will perform {sum(contests_count_list) * [1,2][self.check_positional_bias]} ({' + '.join([f'{c * [1,2][self.check_positional_bias]}' for c in contests_count_list])}) pairwise comparisons"
        )

        response_pairs_list: list[list[list[str]]] = []
        option_pairs_list: list[list[list[str]]] = []
        predictions_names = set(predictions[0].keys())
        for i, combination_indexes in enumerate(combination_indexes_list):
            instance_predictions = predictions[i]
            instance_predictions_names = list(instance_predictions.keys())
            if set(instance_predictions_names) != predictions_names:
                raise Exception(
                    f"The set of prediction names is different between instance 0 and instance {i}. In prediction 0, it is {sorted(predictions_names)}. In prediction {i}, it is {sorted(instance_predictions_names)}. Make sure the same number of predictions is passed for all instances."
                )

            response_pairs: list[list[str]] = []
            option_pairs: list[list[str]] = []
            for combination in combination_indexes:
                (idx_1, idx_2) = combination
                response_name_1 = instance_predictions_names[idx_1]
                response_name_2 = instance_predictions_names[idx_2]
                response_pairs.append(
                    [
                        instance_predictions[response_name_1],
                        instance_predictions[response_name_2],
                    ]
                )
                option_pairs.append([response_name_1, response_name_2])
            response_pairs_list.append(response_pairs)
            option_pairs_list.append(option_pairs)

        criterias = self.get_criterias(task_data, instances_count)
        contexts = self.get_contexts(task_data)
        if self.check_positional_bias:
            criterias.extend(criterias)
            contexts.extend(contexts)
            for response_pairs, option_pairs in zip(
                response_pairs_list, option_pairs_list
            ):
                response_pairs += [
                    list(reversed(response_pair)) for response_pair in response_pairs
                ]
                option_pairs += [
                    list(reversed(option_pair)) for option_pair in option_pairs
                ]

        assessment_instances = [
            {
                "context_variables": contexts[i],
                "response_a": response_pair[0],
                "response_b": response_pair[1],
                "option_a": option_pair[0],
                "option_b": option_pair[1],
                "criteria_name": criterias[i].name,
                "criteria_description": criterias[i].description,
                "data_classification_policy": ["public"],
            }
            for i, (response_pairs, option_pairs) in enumerate(
                zip(response_pairs_list, option_pairs_list)
            )
            for response_pair, option_pair in zip(response_pairs, option_pairs)
        ]
        assessment_prompts, assessment_outputs, _ = self.perform_evaluation_step(
            assessment_instances, self.assessment_task, self.assessment_template
        )
        self.logger.info("The assessment was generated successfully.")

        # the slices used to get the assessment for each summary generation instance
        # it will grab the whole assessment for a particular instance or half of it depending on the value of check_positional_bias
        incremental_contests_count_list = [
            sum(contests_count_list[: i + 1]) for i in range(len(contests_count_list))
        ]

        # Summarisation Stage
        summarization_prompts = None
        summarization_outputs = None
        if self.generate_summaries:
            incremental_contests_count_with_positional_bias_list = [
                incremental_contests_count * [1, 2][self.check_positional_bias]
                for incremental_contests_count in incremental_contests_count_list
            ]
            assessment_for_summaries_slice_list = [
                slice(
                    incremental_contests_count_with_positional_bias_list[i - 1]
                    if i > 0
                    else 0,
                    (
                        incremental_contests_count_with_positional_bias_list[i - 1]
                        if i > 0
                        else 0
                    )
                    + contests_count_list[i],
                )
                for i in range(len(contests_count_list))
            ]
            summarization_instances = [
                {
                    "assessment": assessment_output,
                    "data_classification_policy": ["public"],
                }
                for assessment_for_summaries_slice in assessment_for_summaries_slice_list
                for assessment_output in assessment_outputs[
                    assessment_for_summaries_slice
                ]
            ]

            (
                summarization_prompts,
                summarization_outputs,
                _,
            ) = self.perform_evaluation_step(
                summarization_instances,
                self.summarization_task,
                self.summarization_template,
            )
            self.logger.info("The summary was generated successfully.")

        score_option_instruction_list = [
            "".join(
                [
                    f'Choose "{option}" if Response {option} is better quality.\n'
                    for option in option_pair
                ]
            )
            for option_pairs in option_pairs_list
            for option_pair in option_pairs
        ]

        option_selection_instances = [
            {
                "options": [f"Response {option}" for option in option_pair],
                "score_option_instruction": score_option_instruction,
                "data_classification_policy": ["public"],
            }
            for option_pair, score_option_instruction in zip(
                [
                    option_pair
                    for option_pairs in option_pairs_list
                    for option_pair in option_pairs
                ],
                score_option_instruction_list,
            )
        ]

        previous_messages = [
            [assessment_prompt[0], {"role": "assistant", "content": assessment_output}]
            for assessment_prompt, assessment_output in zip(
                assessment_prompts, assessment_outputs
            )
        ]

        (
            option_selection_prompts,
            option_selection_outputs,
            selections,
        ) = self.perform_evaluation_step(
            option_selection_instances,
            self.option_selection_task,
            self.option_selection_template,
            previous_messages,
        )
        # Selections are of the form 'Response n', so we just keep n
        selections = [selection.split(" ")[-1] for selection in selections]
        self.logger.info("The selections were calculated successfully.")
        results = []
        slice_start = 0
        for i, incremental_contests_count in enumerate(incremental_contests_count_list):
            slice_end = slice_start + contests_count_list[i]
            if self.check_positional_bias:
                slice_end += contests_count_list[i]
            sli = slice(slice_start, slice_end)
            sli_summarization = slice(
                (incremental_contests_count_list[i - 1] if i > 0 else 0),
                (incremental_contests_count_list[i - 1] if i > 0 else 0)
                + incremental_contests_count,
            )
            instance_results = self.get_instance_results(
                predictions[i],
                assessment_prompts[sli],
                assessment_outputs[sli],
                summarization_prompts[sli_summarization]
                if self.generate_summaries
                else None,
                summarization_outputs[sli_summarization]
                if self.generate_summaries
                else None,
                option_selection_prompts[sli],
                option_selection_outputs[sli],
                selections[sli],
                contests_count_list[i],
                combination_indexes_list[i],
                criterias[i]
            )
            results.append(instance_results)
            slice_start = slice_end
        return results
