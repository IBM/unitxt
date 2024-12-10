from unitxt.eval_assist_llm_as_judge import EvalAssistLLMAsJudge

from .api import infer, select
from .artifact import fetch_artifact
from .eval_assist_constants import (
    CriteriaWithOptions,
    OptionSelectionStrategyEnum,
)
from .eval_assist_chat_templates import direct_assessment_template_dict
from .inference import NoInputLogProbsExeption
from .task import Task


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
        elif isinstance(self.criteria, CriteriaWithOptions):
            self.logger.info(
                "Reading criteria from self. Criteria is a single CriteriaWithOptions, replicating it for all predictions"
            )
            criterias: list[CriteriaWithOptions] = [
                self.criteria
            ] * eval_count
        else:
            criterias = self.criteria
        self.logger.info(f"First criteria name is '{criterias[0].name}'")
        return criterias

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

            summarization_outputs_dataset = infer(
                summarization_instances,
                task=self.summarization_task,
                engine=self.inference_engine,
                template=self.summarization_template,
                format=self.format,
                return_data=True,
            )

            summarization_prompts: list[str] = [
                instance["source"] for instance in summarization_outputs_dataset
            ]
            summarization_outputs: list[str] = [
                instance["prediction"] for instance in summarization_outputs_dataset
            ]

            self.logger.info("The summary was generated successfully.")

        selection_instances = [
            {
                "criteria_description": criteria_description,
                "score_option_instruction": score_option_instruction,
                "options": criteria_option_names,
                "data_classification_policy": ["public"],
            }
            for criteria_description, score_option_instruction, criteria_option_names in zip(
                criteria_description_list,
                score_option_instruction_list,
                criteria_option_names_list
            )
        ]

        previous_messages = [[assessment_prompt[0], {'role': 'assistant', 'content': assessment_output}] for assessment_prompt, assessment_output in zip(assessment_prompts, assessment_outputs)]

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
                    previous_messages=previous_messages
                )
                option_selection_prompts: list[str] = [instance["source"] for instance in option_selection_outputs_dataset]
                option_selection_outputs: list[str] = [instance["prediction"] for instance in option_selection_outputs_dataset]
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
                previous_messages=previous_messages
            )
            option_selection_prompts: list[str] = [instance["source"] for instance in option_selection_outputs_dataset]
            option_selection_outputs: list[str] = [instance["raw_prediction"] for instance in option_selection_outputs_dataset]
            selections: list[str] = [instance["prediction"] for instance in option_selection_outputs_dataset]
            
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
        results = [
            {
                key: value
                for key, value in {
                    "score": scores[i],
                    "mapped_score": scores[i],
                    "positional_bias": positional_bias[i] if self.check_positional_bias else None,
                    "selected_option": selections[i],
                    "positional_bias_selected_option": selections[evaluations_count + i] if self.check_positional_bias else None,
                    "assessment": assessment_outputs_dataset[i]["prediction"],
                    "positional_bias_assessment": assessment_outputs_dataset[i + evaluations_count]["prediction"] if self.check_positional_bias else None,
                    "option_selection_prompt": option_selection_prompts[i],
                    "posional_bias_option_selection_prompt": option_selection_prompts[i + evaluations_count],
                    "summary": summarization_outputs[i] if self.generate_summaries else None,
                    "prompts": {
                        "assessment": assessment_prompts[i],
                        "positional_bias_assessment": assessment_prompts[evaluations_count + i],
                        "summarization": summarization_prompts[i] if self.generate_summaries else None,
                        "option_selection": option_selection_prompts[i],
                        "posional_bias_option_selection": option_selection_prompts[i + evaluations_count],
                    },
                    "option_selection_completion": option_selection_outputs[i] if self.option_selection_strategy == OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT else None,
                    "positional_bias_option_selection_completion": option_selection_outputs[evaluations_count + i] if self.option_selection_strategy== OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT else None,
                    "option_selection_strategy": self.option_selection_strategy.name,
                }.items()
                if value is not None
            }
            for i in range(evaluations_count)
        ]

        return results
