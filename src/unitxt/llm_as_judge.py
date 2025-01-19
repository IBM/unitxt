import itertools
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Union

from .api import infer
from .artifact import fetch_artifact
from .dict_utils import dict_get
from .error_utils import UnitxtError
from .inference import (
    InferenceEngine,
    OptionSelectingByLogProbsInferenceEngine,
)
from .llm_as_judge_chat_templates import direct_template_dict, pairwise_template_dict
from .llm_as_judge_constants import (
    DIRECT_CRITERIAS,
    EVALUATOR_TO_MODEL_ID,
    EVALUATORS_METADATA,
    INFERENCE_ENGINE_NAME_TO_CLASS,
    MODEL_RENAMINGS,
    PAIRWISE_CRITERIAS,
    Criteria,
    CriteriaOption,
    CriteriaWithOptions,
    DirectCriteriaCatalogEnum,
    EvaluatorMetadata,
    EvaluatorNameEnum,
    EvaluatorTypeEnum,
    ModelProviderEnum,
    PairwiseCriteriaCatalogEnum,
)
from .llm_as_judge_from_template import LLMAsJudge, LLMAsJudgeBase, TaskBasedLLMasJudge
from .llm_as_judge_operators import (
    CreateCriteriaFromDict,
    CreateCriteriaFromJson,
    CreateCriteriaFromString,
    CreateCriteriaWithOptionsFromDict,
    CreateCriteriaWithOptionsFromJson,
    CreateYesNoCriteriaFromString,
    CreateYesNoPartiallyCriteriaFromString,
    LoadCriteria,
    LoadCriteriaWithOptions,
)
from .llm_as_judge_utils import (
    get_evaluator_metadata,
    get_parsed_context,
    rank_indexes,
    rename_model_if_required,
)
from .logging_utils import get_logger
from .metrics import BulkInstanceMetric
from .task import Task
from .templates import Template


class LLMJudge(BulkInstanceMetric):
    inference_engine: InferenceEngine
    # option_selection_strategy: OptionSelectionStrategyEnum = (
    #     OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT
    # )
    evaluator_name: EvaluatorNameEnum = None
    check_positional_bias: bool = True
    context_fields: Union[str, List[str], Dict[str, str]] = ["context"]
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
        if isinstance(self.context_fields, List):
            self.context_fields = {
                context_field: context_field for context_field in self.context_fields
            }

        if self.evaluator_name is None:
            self.evaluator_name = self.inference_engine.get_engine_id()
        elif not isinstance(self.evaluator_name, EvaluatorNameEnum):
            self.evaluator_name = EvaluatorNameEnum[self.evaluator_name]

    def before_process_multi_stream(self):
        super().before_process_multi_stream()
        # We check the criteria here and not in verify(), because we want catalog
        # may contain a partially initialized object, and verify() method
        # is called when creating the object and not when using it.
        if self.criteria is None and self.criteria_field is None:
            raise UnitxtError(
                f"You must set either the 'criteria' field of the {__class__.__name__} metric to define one criteria to evaluate on all instance, or set a 'criteria_field' of the metric to evaluate on each instance based on the criteria specified in that field of each instance."
            )
        return

    def get_contexts(self, task_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        return [
            get_parsed_context(
                {
                    context_field_name: dict_get(td, context_field)
                    for context_field_name, context_field in self.context_fields.items()
                }
            )
            for td in task_data
        ]

    def perform_evaluation_step(
        self,
        instances: list,
        task: Task,
        template: Template,
        previous_messages: Optional[List[Dict[str, str]]] = None,
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
        prompts: List[str] = [instance["source"] for instance in outputs_dataset]
        raw_predictions: List[str] = [
            instance["raw_prediction"] for instance in outputs_dataset
        ]
        predictions: List[str] = [
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

    def get_criterias(self, task_data, eval_count):
        if self.criteria is None:
            if self.criteria_field not in task_data[0]:
                raise UnitxtError(
                    f"The criteria field `{self.criteria_field}` required for {__class__.__name__} is not found in instance.  Perhaps you meant '{get_close_matches(self.criteria_field, task_data[0].keys(), n=1, cutoff=0.0)[0]}'?"
                )
            self.logger.info(
                f"Reading criteria from the task_data field '{self.criteria_field}'"
            )
            criterias = [
                fetch_artifact(task_data_instance[self.criteria_field])[0]
                for task_data_instance in task_data
            ]
        else:
            self.logger.info(
                "Reading criteria from self. Criteria is a single CriteriaWithOptions, replicating it for all predictions"
            )
            criterias: List[Criteria] = [self.criteria] * eval_count
        unique_criteria_names = list({criteria.name for criteria in criterias})

        self.logger.info(f"Criteria names are '{', '.join(unique_criteria_names)}'")
        return criterias


class LLMJudgeDirect(LLMJudge):
    criteria: CriteriaWithOptions = None
    main_score = "llm_as_judge"
    reduction_map = {"mean": ["llm_as_judge"]}

    def prepare(self):
        super().prepare()
        self.assessment_template = direct_template_dict["assessment"]
        self.summarization_template = direct_template_dict["summarization"]
        self.option_selection_template = direct_template_dict["answer"]

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

    def before_process_multi_stream(self):
        super().before_process_multi_stream()
        if self.criteria is not None and not isinstance(
            self.criteria, CriteriaWithOptions
        ):
            raise Exception(
                f"The type of the criteria must be 'CriteriaWithOptions', instead it is of type '{type(self.criteria)}'"
            )
        return

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

    def set_main_score(self, criterias: List[CriteriaWithOptions]):
        unique_criteria_names = list({criteria.name for criteria in criterias})
        if len(unique_criteria_names) == 1 and criterias[0].name != "":
            self.main_score = "_".join(criterias[0].name.lower().split(" "))
            self.reduction_map = {"mean": [self.main_score]}

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
        criterias: List[CriteriaWithOptions],
    ) -> List[Dict[str, Any]]:
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

        results = [
            {
                self.main_score: scores[i],
                f"using_{self.evaluator_name.lower()}_{self.inference_engine.label}": scores[
                    i
                ],
                "positional_bias": positional_bias[i]
                if self.check_positional_bias
                else None,
                "selected_option": selections[i],
                "positional_bias_selected_option": selections[evaluations_count + i]
                if self.check_positional_bias
                else None,
                "assessment": assessment_outputs[i],
                "positional_bias_assessment": assessment_outputs[i + evaluations_count]
                if self.check_positional_bias
                else None,
                "summary": summarization_outputs[i]
                if self.generate_summaries
                else None,
                "prompts": {
                    "assessment": assessment_prompts[i],
                    "positional_bias_assessment": assessment_prompts[
                        evaluations_count + i
                    ]
                    if self.check_positional_bias
                    else None,
                    "summarization": summarization_prompts[i]
                    if self.generate_summaries
                    else None,
                    "option_selection": option_selection_prompts[i],
                    "posional_bias_option_selection": option_selection_prompts[
                        i + evaluations_count
                    ]
                    if self.check_positional_bias
                    else None,
                }
                if self.include_prompts_in_result
                else None,
                "option_selection_completion": option_selection_outputs[i],
                "positional_bias_option_selection_completion": option_selection_outputs[
                    evaluations_count + i
                ]
                if self.check_positional_bias
                else None,
                "criteria": criterias[i].to_json(),
            }
            for i in range(evaluations_count)
        ]
        # add main_score to each result
        return [
            {
                f"{self.main_score}_{k}" if k != self.main_score else self.main_score: v
                for k, v in r.items()
            }
            for r in results
        ]

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict[str, Any]],
    ) -> dict:
        self.logger.info(
            f'Starting evaluation with evaluator "{self.evaluator_name}" and provider "{self.inference_engine.get_pretty_print_name()}'
        )
        evaluations_count = len(predictions)
        # TODO: find out how to serialize and deserialize enums
        criterias = self.get_criterias(task_data, evaluations_count)
        self.set_main_score(criterias)
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

        summarization_prompts = None
        summarization_outputs = None
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

        results = self.get_results(
            assessment_prompts,
            assessment_outputs,
            summarization_prompts,
            summarization_outputs,
            option_selection_prompts,
            option_selection_outputs,
            selections,
            evaluations_count,
            criterias,
        )
        return self.clean_results(results)


class LLMJudgePairwise(LLMJudge):
    reduction_map = {"mean": ["score"]}
    main_score = "1_winrate"
    prediction_type = List[str]

    def prepare(self):
        super().prepare()
        self.assessment_template = pairwise_template_dict["assessment"]
        self.summarization_template = pairwise_template_dict["summarization"]
        self.option_selection_template = pairwise_template_dict["answer"]

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

    def before_process_multi_stream(self):
        super().before_process_multi_stream()
        if self.criteria is not None and not isinstance(self.criteria, Criteria):
            raise Exception(
                f"The type of the criteria must be 'Criteria', instead it is of type '{type(self.criteria)}'"
            )
        return

    def get_instance_results(
        self,
        instance_predictions: Dict[str, str],
        assessment_prompts,
        assessment_outputs,
        summarization_prompts,
        summarization_outputs,
        option_selection_prompts,
        option_selection_outputs,
        selections,
        contests_count,
        combination_indexes,
        criteria: Criteria,
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
                per_response_results[response_name_1][
                    "positional_bias_assessments"
                ].append(assessment_outputs[positional_bias_i])
                per_response_results[response_name_2][
                    "positional_bias_assessments"
                ].append(assessment_outputs[positional_bias_i])
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

            per_response_results[response_name_1]["option_selection_outputs"].append(
                option_selection_outputs[i]
            )
            per_response_results[response_name_2]["option_selection_outputs"].append(
                option_selection_outputs[i]
            )
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
            per_response_results[key]["llm_as_judge"] = winrate
        # calculate ranking
        ranking = rank_indexes(
            [result["winrate"] for result in per_response_results.values()]
        )

        for response_name, r_i in zip(response_names, ranking):
            per_response_results[response_name]["ranking"] = r_i + 1

        for response_name in response_names:
            # add response name
            per_response_results[response_name]["response_name"] = response_name

        all_results = {}
        for response_name in response_names:
            single_result = per_response_results[response_name]
            for metric in single_result.keys():
                all_results[f"{response_name}_{metric}"] = single_result[metric]

        all_results["criteria"] = criteria.to_json()
        return self.clean_results(all_results)

    def parse_prediction_to_dict(self, prediction: Union[Dict[str, str], List[str]]):
        if isinstance(prediction, list):
            return {f"{key + 1}": value for key, value in enumerate(prediction)}

        raise Exception(
            f"Prediction may be a list or a dict. Instead got type {type(prediction)}"
        )

    def convert_predictions_to_dicts(
        self, predictions: Union[List[Dict[str, str]], List[str]]
    ):
        return [self.parse_prediction_to_dict(prediction) for prediction in predictions]

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict[str, str]],
    ) -> dict:
        self.logger.info(
            f'Starting evaluation with evaluator "{self.evaluator_name}" and provider {self.inference_engine.get_pretty_print_name()}'
        )
        predictions = self.convert_predictions_to_dicts(predictions)
        instances_count = len(predictions)
        self.reduction_map = {"mean": ["score"]}
        self.reduction_map["mean"].extend(
            [f"{key}_winrate" for key in predictions[0].keys()]
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

        response_pairs_list: List[List[List[str]]] = []
        option_pairs_list: List[List[List[str]]] = []
        predictions_names = set(predictions[0].keys())
        for i, combination_indexes in enumerate(combination_indexes_list):
            instance_predictions = predictions[i]
            instance_predictions_names = list(instance_predictions.keys())
            if set(instance_predictions_names) != predictions_names:
                raise Exception(
                    f"The set of prediction names is different between instance 0 and instance {i}. In prediction 0, it is {sorted(predictions_names)}. In prediction {i}, it is {sorted(instance_predictions_names)}. Make sure the same number of predictions is passed for all instances."
                )

            response_pairs: List[List[str]] = []
            option_pairs: List[List[str]] = []
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
                criterias[i],
            )
            results.append(instance_results)
            slice_start = slice_end

        return results
