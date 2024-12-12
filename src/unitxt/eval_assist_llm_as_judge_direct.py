from typing import Optional

from .api import infer
from .artifact import fetch_artifact
from .eval_assist_chat_templates import direct_assessment_template_dict
from .eval_assist_constants import (
    CriteriaOption,
    CriteriaWithOptions,
    OptionSelectionStrategyEnum,
)
from .eval_assist_llm_as_judge import EvalAssistLLMAsJudge
from .task import Task
from .templates import Template


class EvalAssistLLMAsJudgeDirect(EvalAssistLLMAsJudge):
    criteria: CriteriaWithOptions = None
    reduction_map = {"mean": ["score"]}
    main_score = "score"

    def prepare(self):
        super().prepare()
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
                "criteria_description": str,
                "score_option_instruction": str,
                "options": list,
            },
            reference_fields={},
            prediction_type=str,
            metrics=[],
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

    def get_criterias(self, task_data, eval_count):
        if self.criteria is None:
            self.logger.info("Reading criteria from the task_data")
            criterias = []
            for task_data_criteria in [
                task_data_instance["criteria"] for task_data_instance in task_data
            ]:
                if isinstance(task_data_criteria, dict):
                    try:
                        criteria = CriteriaWithOptions(
                            name=task_data_criteria["name"],
                            description=task_data_criteria["description"],
                            options=[
                                CriteriaOption(
                                    name=option_dict["name"],
                                    description=option_dict["description"],
                                )
                                for option_dict in task_data_criteria["options"]
                            ],
                        )
                    except:
                        raise Exception(
                            "The criteria dict couldn't be parsed correctly. It has to have the json representation of CriteriaWithOptions"
                        ) from None
                elif isinstance(task_data_criteria, str):
                    try:
                        criteria = fetch_artifact(task_data_criteria)[0]
                    except Exception:
                        # Criteria can't be converted into an artifact
                        # Use it a the description of a Yes/No criteria
                        criteria = CriteriaWithOptions(
                            name=f"Unknown ({task_data_criteria[:20]}...)",
                            description=task_data_criteria,
                            options=[
                                CriteriaOption(name="Yes", description=""),
                                CriteriaOption(name="No", description=""),
                            ],
                            option_map={
                                "Yes": 1.0,
                                "No": 0.0,
                            },
                        )
                else:
                    raise Exception(
                        f"The criteria needs to be of type str (criteria catalog name or yes/no-criteria description) or dict (criteria json representation). Instead, it is of type {type(task_data_criteria)}"
                    )
                criterias.append(criteria)
        else:
            self.logger.info(
                "Reading criteria from self. Criteria is a single CriteriaWithOptions, replicating it for all predictions"
            )
            if not isinstance(self.criteria, CriteriaWithOptions):
                raise Exception(
                    f"The type of the criteria must be 'CriteriaWithOptions', instead it is of type '{type(self.criteria)}'"
                )

            criterias: list[CriteriaWithOptions] = [self.criteria] * eval_count

        self.logger.info(
            f"Criteria names are '{', '.join([criteria.name for criteria in list(set(criterias))])}'"
        )
        return criterias

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

    def compute(
        self,
        references: list[list[str]],
        predictions: list[str],
        task_data: list[dict[str, any]],
    ) -> dict:
        self.logger.info(
            f'Starting evaluation with evaluator "{self.evaluator_name}" and provider "{self.inference_engine.get_pretty_print_name()}" with strategy "{self.option_selection_strategy.name}"'
        )

        evaluations_count = len(predictions)
        # TODO: find out how to serialize and deserialize enums
        criterias = self.get_criterias(task_data, evaluations_count)
        contexts = self.get_contexts(task_data)
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

        parsed_criterias = [
            self.get_parsed_criteria(criteria) for criteria in criterias
        ]

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
        assessment_prompts, assessment_outputs, _ = self.perform_evaluation_step(
            assessment_instances, self.assessment_task, self.assessment_template
        )
        self.logger.info("The assessment was generated successfully.")

        if self.generate_summaries:
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

        option_selection_instances = [
            {
                "criteria_description": criteria_description,
                "score_option_instruction": score_option_instruction,
                "options": criteria_option_names,
                "data_classification_policy": ["public"],
            }
            for criteria_description, score_option_instruction, criteria_option_names in zip(
                criteria_description_list,
                score_option_instruction_list,
                criteria_option_names_list,
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
        #             judgement_instances,
        #             engine=self.inference_engine,
        #             task=self.option_selection_task,
        #             template=self.option_selection_template,
        #             format=self.format,
        #             return_data=True,
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
                    "llm_as_a_judge_score": scores[i],
                    "positional_bias": positional_bias[i]
                    if self.check_positional_bias
                    else None,
                    "selected_option": selections[i],
                    "positional_bias_selected_option": selections[evaluations_count + i]
                    if self.check_positional_bias
                    else None,
                    "assessment": assessment_outputs[i],
                    "positional_bias_assessment": assessment_outputs[
                        i + evaluations_count
                    ]
                    if self.check_positional_bias
                    else None,
                    "summary": summarization_outputs[i]
                    if self.generate_summaries
                    else None,
                    "prompts": {
                        "assessment": assessment_prompts[i],
                        "positional_bias_assessment": assessment_prompts[
                            evaluations_count + i
                        ],
                        "summarization": summarization_prompts[i]
                        if self.generate_summaries
                        else None,
                        "option_selection": option_selection_prompts[i],
                        "posional_bias_option_selection": option_selection_prompts[
                            i + evaluations_count
                        ],
                    },
                    "option_selection_completion": option_selection_outputs[i]
                    if self.option_selection_strategy
                    == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT
                    else None,
                    "positional_bias_option_selection_completion": option_selection_outputs[
                        evaluations_count + i
                    ]
                    if self.option_selection_strategy
                    == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT
                    else None,
                    "option_selection_strategy": self.option_selection_strategy.name,
                    "criteria_name": criterias[i].name,
                }.items()
                if value is not None
            }
            for i in range(evaluations_count)
        ]
