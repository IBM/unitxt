import itertools

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
    reduction_map = {"mean": ["winrate"]}
    main_score = "winrate"

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
                "choose_response_instruction": str,
                "score_option_instruction": str,
                "options": list,
            },
            reference_fields={},
            prediction_type=str,
            metrics=[],
        )

    def get_criterias(self, task_data, eval_count):
        if self.criteria is None:
            for criteria_dict in [
                task_data_instance["criteria"] for task_data_instance in task_data
            ]:
                if not isinstance(criteria_dict, dict):
                    raise Exception(
                        f"The type of the criteria must be of type dict, instead it is of type '{type(criteria_dict)}'"
                    )

            criterias = [
                Criteria(
                    name=criteria_dict["name"],
                    description=criteria_dict["description"],
                )
                for criteria_dict in [
                    task_data_instance["criteria"] for task_data_instance in task_data
                ]
            ]
        # criteria is in passes in the constructor
        else:
            self.logger.info(
                "Reading criteria from self. Criteria is a single Criteria, replicating it for all predictions"
            )
            if not isinstance(self.criteria, Criteria):
                raise Exception(
                    f"The type of the criteria must be 'Criteria', instead it is of type '{type(self.criteria)}'"
                )

            criterias: list[Criteria] = [self.criteria] * eval_count

        self.logger.info(f"First criteria name is '{criterias[0].name}'")
        return criterias

    def get_results(
        self,
        assessment_prompts,
        assessment_outputs,
        summarization_prompts,
        summarization_outputs,
        option_selection_prompts,
        option_selection_outputs,
        selections,
        evaluations_count,
        contests_count,
        combination_indexes,
    ):
        per_response_results = {
            f"{i+1}": {
                "summaries": [],
                "contest_results": [],
                "selections": [],
                "compared_to": [],
                "assessments": [],
                "positional_bias_assessments": [],
                "option_selection_outputs": [],
                "positional_bias": [],
                "positional_bias_option_selection_outputs": [],
                "prompts": {
                    "assessment": [],
                    "positional_bias_assessment": [],
                    "option_selection": [],
                    "positional_bias_option_selection": [],
                    "summary": [],
                },
            }
            for i in range(evaluations_count)
        }

        ### process results
        positional_bias = None
        if self.check_positional_bias:
            positional_bias = [
                selections[i] != selections[contests_count + i]
                for i in range(contests_count)
            ]
        for i in range(contests_count):
            (idx_1, idx_2) = combination_indexes[i]
            response_name_1 = f"{idx_1+1}"
            response_name_2 = f"{idx_2+1}"
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
            per_response_results[response_name_1]["positional_bias_assessments"].append(
                assessment_outputs[contests_count + i]
            )
            per_response_results[response_name_1]["selections"].append(
                selected_response_name
            )
            per_response_results[response_name_2]["selections"].append(
                selected_response_name
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
            per_response_results[response_name_1]["prompts"]["assessment"].append(
                assessment_prompts[i]
            )
            per_response_results[response_name_2]["prompts"]["assessment"].append(
                assessment_prompts[i]
            )
            if self.generate_summaries:
                # add summaries
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

            per_response_results[response_name_1]["prompts"]["option_selection"].append(
                option_selection_prompts[i]
            )
            per_response_results[response_name_2]["prompts"]["option_selection"].append(
                option_selection_prompts[i]
            )

            ## add positional bias
            if self.check_positional_bias:
                per_response_results[response_name_1]["positional_bias"].append(
                    positional_bias[i]
                )
                per_response_results[response_name_2]["positional_bias"].append(
                    positional_bias[i]
                )

                # add prompts
                per_response_results[response_name_1]["prompts"][
                    "positional_bias_assessment"
                ].append(assessment_prompts[contests_count + i])
                per_response_results[response_name_2]["prompts"][
                    "positional_bias_assessment"
                ].append(assessment_prompts[contests_count + i])
                per_response_results[response_name_1]["prompts"][
                    "positional_bias_option_selection"
                ].append(option_selection_prompts[contests_count + i])
                per_response_results[response_name_2]["prompts"][
                    "positional_bias_option_selection"
                ].append(option_selection_prompts[contests_count + i])

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
                        "positional_bias_option_selection_outputs"
                    ].append(option_selection_outputs[contests_count + i])
                    per_response_results[response_name_2][
                        "positional_bias_option_selection_outputs"
                    ].append(option_selection_outputs[contests_count + i])

        # add winrate
        for key in per_response_results:
            contest_results = per_response_results[key]["contest_results"]
            winrate = sum(contest_results) / len(contest_results)
            per_response_results[key]["winrate"] = winrate
            per_response_results[key]["llm_as_a_judge_score"] = winrate
        # calculate ranking
        ranking = rank_indexes(
            [result["winrate"] for result in per_response_results.values()]
        )

        for i, key in enumerate(per_response_results.keys()):
            per_response_results[key]["ranking"] = ranking[i] + 1

        for key in per_response_results:
            # add response name
            per_response_results[key]["response_name"] = key

        return per_response_results

    def compute(
        self,
        references: list[list[str]],
        predictions: list[str],
        task_data: list[dict[str, str]],
    ) -> dict:
        self.logger.info(
            f'Starting evaluation with evaluator "{self.evaluator_name}" and provider {self.inference_engine.get_pretty_print_name()}'
        )
        evaluations_count = len(predictions)
        combination_indexes = list(itertools.combinations(range(evaluations_count), 2))
        contests_count = len(combination_indexes)
        self.logger.info(
            f"The evaluation will perform {contests_count} pairwise comparisons"
        )

        response_pairs: list[list[str]] = []
        option_pairs: list[list[str]] = []
        for combination in combination_indexes:
            (response_name_1, response_name_2) = combination
            response_pairs.append(
                [predictions[response_name_1], predictions[response_name_2]]
            )
            option_pairs.append([f"{response_name_1 + 1}", f"{response_name_2 + 1}"])

        task_data = [task_data[0]] * contests_count

        criterias = self.get_criterias(task_data, contests_count)
        contexts = self.get_contexts(task_data)

        if self.check_positional_bias:
            criterias += criterias
            contexts += contexts
            predictions += predictions
            response_pairs += [
                list(reversed(response_pair)) for response_pair in response_pairs
            ]
            option_pairs += [
                list(reversed(option_pair)) for option_pair in option_pairs
            ]

        assessment_for_summaries_slice = slice(0, contests_count)

        assessment_instances = [
            {
                "context_variables": context,
                "response_a": response_pair[0],
                "response_b": response_pair[1],
                "option_a": option_pair[0],
                "option_b": option_pair[1],
                "criteria_name": criteria.name,
                "criteria_description": criteria.description,
                "data_classification_policy": ["public"],
            }
            for context, criteria, response_pair, option_pair in zip(
                contexts, criterias, response_pairs, option_pairs
            )
        ]
        assessment_prompts, assessment_outputs, _ = self.perform_evaluation_step(
            assessment_instances, self.assessment_task, self.assessment_template
        )
        self.logger.info("The assessment was generated successfully.")

        # Summarisation Stage
        summarization_prompts = None
        summarization_outputs = None
        if self.generate_summaries:
            summarization_instances = [
                {
                    "assessment": assessment_output,
                    "data_classification_policy": ["public"],
                }
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

            self.logger.info("The summary was generated successfully.")

        choose_response_instructions = [
            "".join(
                [
                    f'Choose "{option}" if Response {option} is better quality.\n'
                    for option in option_pair
                ]
            )
            for option_pair in option_pairs
        ]
        score_option_instruction_list = [
            "".join(
                [
                    f'Choose "{option}" if Response {option} is better quality.\n'
                    for option in option_pair
                ]
            )
            for option_pair in option_pairs
        ]
        option_selection_instances = [
            {
                "choose_response_instruction": choose_response_instruction,
                "options": [f"Response {option}" for option in option_pair],
                "score_option_instruction": score_option_instruction,
                "data_classification_policy": ["public"],
            }
            for choose_response_instruction, option_pair, score_option_instruction in zip(
                choose_response_instructions,
                option_pairs,
                score_option_instruction_list,
            )
        ]
        previous_messages = [
            [assessment_prompt[0], {"role": "assistant", "content": assessment_output}]
            for assessment_prompt, assessment_output in zip(
                assessment_prompts, assessment_outputs
            )
        ]

        # parse_output_logprobs_failed = False
        # if (
        #     self.option_selection_strategy
        #     == OptionSelectionStrategyEnum.PARSE_OPTION_LOGPROB
        # ):
        #     try:
        #         option_selection_outputs_dataset = select(
        #             option_selection_instances,
        #             engine=self.inference_engine,
        #             task=self.option_selection_task,
        #             template=self.option_selection_template,
        #             return_data=True,
        #             format=self.format,
        #             previous_messages=previous_messages,
        #         )
        #         option_selection_prompts: list[str] = [
        #             instance["source"] for instance in option_selection_outputs_dataset
        #         ]
        #         option_selection_outputs: list[str] = [
        #             instance["prediction"]
        #             for instance in option_selection_outputs_dataset
        #         ]
        #         selections = option_selection_outputs
        #         # take the number of the response from 'Response n'
        #     except NoInputLogProbsError as e:
        #         self.logger.error(f"An error occurred: {e}")
        #         self.logger.warning(
        #             f"{self.option_selection_strategy.name} failed. trying {OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT.name} instead."
        #         )
        #         parse_output_logprobs_failed = True

        # if (
        #     self.option_selection_strategy
        #     == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT
        #     or parse_output_logprobs_failed
        # ):
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

        per_response_results = self.get_results(
            assessment_prompts,
            assessment_outputs,
            summarization_prompts,
            summarization_outputs,
            option_selection_prompts,
            option_selection_outputs,
            selections,
            evaluations_count,
            contests_count,
            combination_indexes,
        )
        # remove None values from the result dict, e.g. when positional_bias_check is False there is no positional_bias entry in the dict
        return [
            {
                metric_key: value
                for metric_key, value in per_response_results[key].items()
                if value is not None or (isinstance(value, list) and len(value) > 0)
            }
            for key in [f"{i+1}" for i in range(evaluations_count)]
        ]
