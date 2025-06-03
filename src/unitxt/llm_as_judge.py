import itertools
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Union

from .api import infer
from .artifact import fetch_artifact
from .dict_utils import dict_get
from .error_utils import UnitxtError
from .inference import (
    InferenceEngine,
)
from .llm_as_judge_chat_templates import direct_template_dict, pairwise_template_dict
from .llm_as_judge_constants import (
    DIRECT_CRITERIA,
    EVALUATOR_TO_MODEL_ID,
    EVALUATORS_METADATA,
    PAIRWISE_CRITERIA,
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
)
from .logging_utils import get_logger
from .metrics import BulkInstanceMetric
from .task import Task
from .templates import Template

logger = get_logger(__name__)


class LLMJudge(BulkInstanceMetric):
    """A metric class to evaluate instances using LLM as a Judge.

    Evaluations are performed in two steps. First, the LLM is asked to generate an assessment following a CoT approach based on the criteria. Then, the same LLM is asked to select one of the available options. A summary of the general assessment can be generated for easy consumption by end users.
    """

    inference_engine: InferenceEngine
    """The engine used for generating predictions in the different evaluation steps."""

    evaluator_name: EvaluatorNameEnum = None
    """The name of the evaluator. It is used for score naming. If not provided `self.inference_engine.get_engine_id()` is used."""

    check_positional_bias: bool = True
    """Flag to check for positional bias. Detecting for positional bias duplicates the amount of inference calls."""

    context_fields: Union[str, List[str], Dict[str, str]] = ["context"]
    """Fields to be used as context. If a dict is provided, the keys are used as the final names in the prompts, while the values are used to access the context variable values in the `task_data` object."""

    generate_summaries: bool = False
    """Flag to generate summaries of the assessments. Defaults to `False`."""

    format: str = "formats.chat_api"
    """The format used for the inference. Defaults to `formats.chat_api` (only allowed value)."""

    include_prompts_in_result: bool = True
    """Flag to include prompts in the result. Defaults to `True`."""

    criteria_field: str = None
    """The field specifying the evaluation criteria in the `task_data` object."""

    criteria: Criteria = None
    """The criteria used for evaluation. If the `criteria_field` is provided, it will take precedence."""

    def prepare(self):
        """Prepares the `LLMJudge` instance by setting up context fields and evaluator name."""
        super().prepare()
        if isinstance(self.context_fields, str):
            self.context_fields = [self.context_fields]
        if isinstance(self.context_fields, List):
            self.context_fields = {
                context_field: context_field for context_field in self.context_fields
            }

        if self.evaluator_name is None:
            self.evaluator_name = self.inference_engine.get_engine_id()

    def before_process_multi_stream(self):
        """Checks the criteria-related fields correctness before processing multiple streams.

        Raises:
            UnitxtError: If both 'criteria' and 'criteria_field' are not set.
        """
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
        """Extracts and parses context fields from task data.

        Args:
            task_data (List[Dict[str, Any]]): The task data containing context information.

        Returns:
            List[Dict[str, str]]: A list of parsed context dictionaries.
        """
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
        """Performs an evaluation step by generating predictions for the given instances.

        Args:
            instances (list): The list of instances to evaluate.
            task (Task): The task associated with the instances.
            template (Template): The template used for generating predictions.
            previous_messages (Optional[List[Dict[str, str]]]): Previous messages for context.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing prompts, raw predictions, and processed predictions. Raw predictions differ from processed predictions only in the completion step, where the processors.match_closest_option is used.
        """
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
        """Cleans the results by removing `None` values and empty lists and dictionaries.

        Args:
            results (Union[dict, list]): The results to clean.

        Returns:
            Union[dict, list]: The cleaned results.
        """
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

    def get_criteria(self, task_data, eval_count):
        """Retrieves the evaluation criteria from the `criteria_field` or from `self`.

        Args:
            task_data (List[Dict[str, Any]]): The task data containing criteria information.
            eval_count (int): The number of evaluations to perform.

        Returns:
            List[Criteria]: A list of criteria for evaluation.

        Raises:
            UnitxtError: If the criteria field is not found in the task data.
        """
        if self.criteria is None:
            if self.criteria_field not in task_data[0]:
                raise UnitxtError(
                    f"The criteria field `{self.criteria_field}` required for {__class__.__name__} is not found in instance.  Perhaps you meant '{get_close_matches(self.criteria_field, task_data[0].keys(), n=1, cutoff=0.0)[0]}'?"
                )
            logger.info(
                f"Reading criteria from the task_data field '{self.criteria_field}'"
            )
            criterias = [
                fetch_artifact(task_data_instance[self.criteria_field])[0]
                for task_data_instance in task_data
            ]
        else:
            logger.info(
                "Reading criteria from self. Criteria is a single CriteriaWithOptions, replicating it for all predictions"
            )
            criterias: List[Criteria] = [self.criteria] * eval_count
        unique_criteria_names = list({criteria.name for criteria in criterias})

        logger.info(f"Criteria names are '{', '.join(unique_criteria_names)}'")
        return criterias


class LLMJudgeDirect(LLMJudge):
    """LLMJudgeDirect is a specialized evaluation metric that performs Direct Assessment using an LLM to score responses based on a predefined evaluation criteria.

    Direct Assessment is an evaluation paradigm in which the LLM selects one of a
    predefined set of options based on an assessment criterion. This approach can
    be used for Likert-scale scoring (e.g., 1-5) or selecting from semantically
    conditioned literals (e.g., Yes/No, Pass/Fail).
    """

    criteria: CriteriaWithOptions = None
    """The evaluation criteria, including a name, description, a predefined set of options and and option_map."""
    main_score = "llm_as_judge"
    """The primary score name used in the results. By default, it will take the value of the criteria name (if only one criteria is being used for evaluation) or "llm_as_judge" otherwise."""
    reduction_map = {"mean": ["llm_as_judge"]}
    """A mapping used for score aggregation. By default, it will take the value of ``{'mean': [<default_main_score_name>]}`` ."""

    def prepare(self):
        super().prepare()
        self.assessment_template = direct_template_dict["assessment"]
        self.summarization_template = direct_template_dict["summarization"]
        self.option_selection_template = direct_template_dict["answer"]

        self.assessment_task = Task(
            input_fields={
                "context_variables": str,
                "response": Any,
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
                "display_options_instruction": str,
                "options": list,
            },
            reference_fields={},
            prediction_type=str,
            metrics=[],
        )

    def before_process_multi_stream(self):
        """Ensures that the criteria is of type `CriteriaWithOptions`, raising an exception otherwise."""
        super().before_process_multi_stream()
        if self.criteria is not None and not isinstance(
            self.criteria, CriteriaWithOptions
        ):
            raise Exception(
                f"The type of the criteria must be 'CriteriaWithOptions', instead it is of type '{type(self.criteria)}'"
            )
        return

    def __get_parsed_criteria(self, criteria: CriteriaWithOptions):
        """Extracts key information from the given criteria.

        Args:
            criteria (CriteriaWithOptions): The evaluation criteria.

        Returns:
            Tuple[str, List[str], str, str]:
            - Criteria description.
            - List of option names.
            - Formatted instruction for displaying options.
            - Instruction for scoring options.
        """
        criteria_description = criteria.description
        criteria_option_names = [o.name for o in criteria.options]

        display_options_instruction = "Choose an option:\n" + "\n".join(
            [
                f'- "{o.name}"{f" if {o.description}" if o.description != "" else ""}'
                for o in criteria.options
            ]
        )

        return (
            criteria_description,
            criteria_option_names,
            display_options_instruction,
        )

    def __set_main_score(self, criterias: List[CriteriaWithOptions]):
        unique_criteria_names = list({criteria.name for criteria in criterias})
        if len(unique_criteria_names) == 1 and criterias[0].name != "":
            self.main_score = "_".join(criterias[0].name.lower().split(" "))
            self.reduction_map = {"mean": [self.main_score]}

    def __get_results(
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
                "positional_bias_summary": summarization_outputs[i]
                if self.generate_summaries and self.check_positional_bias
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
    ) -> List[Dict]:
        r"""Performs direct assessment evaluation on the given predictions and references.

        This method evaluates the quality of of the predictions by calculating scores for each instance based on a criterion.

        Returns:
        --------
        List[Dict]
            A list of dictionaries containing the evaluation results for each instance. The results include the computed scores for each prediction. Each result will have the `score_name` as a prefix, which may be the criterion name if only one used, or "llm_as_judge" if several criteria were used.

            Explanation of fields:

            - `score`: a float representing the evaluation score for the response. The value is calculated from criteria.option_map[selected_option].
            - `using_<evaluator_name>`: Equal to score.
            - `positional_bias`: Boolean indicating whether the assessment detected positional bias. Its final value is selected_option != positional_bias_selected_option
            - `selected_option`: The criteria option that the evaluator chose (e.g., "Could be Improved"). It is calculated by processing `option_selection_completion` using `processors.match_closest_option`
            - `positional_bias_selected_option`: The criteria option that the evaluator chose when checking positional bias.
            - `assessment`: The inference engine's generated text using the `prompts.assessment` prompt.
            - `positional_bias_assessment`: The inference engine's generated text using the `prompts.positional_bias_assessment` prompt.
            - `summary`: An LLM-generated summary of the assessment.
            - `positional_bias_summary`: A LLM-generated summary of the positional bias assessment.
            - `prompts`: A dictionary of prompts used in different stages of evaluation.
                - `assessment`: The prompt used to instruct the model on how to assess the response.
                - `positional_bias_assessment`: The prompt used to instruct the model on how to assess the response in the positional bias check.
                - `summarization`: The prompt used to generate summary of the assessment.
                - `option_selection`: The prompt used to generate a final judgement.
                - `positional_bias_option_selection`: The prompt used to generate a final judgement in the positional bias check.
            - `option_selection_completion`: The inference engine's generated text using `prompts.option_selection`.
            - `positional_bias_option_selection_completion`: The inference engine's generated text using `prompts.positional_bias_option_selection`.
            - `criteria`: A JSON-like string representing the evaluation criteria's artifact.

            Result example:

            .. code-block:: python

                [
                    {
                        "answer_relevance": 1,
                        "answer_relevance_using_granite3.0-2b_litellm": 1,
                        "answer_relevance_positional_bias": false,
                        "answer_relevance_selected_option": "Could be Improved",
                        "answer_relevance_positional_bias_selected_option": "Could be Improved",
                        "answer_relevance_assessment": "To assess the quality of the response, l...",
                        "answer_relevance_positional_bias_assessment": "To assess the quality of the response, l...",
                        "answer_relevance_summary": "A response about apprenticeships during ...",
                        "answer_relevance_positional_bias_summary": "A response about apprenticeships during ...",
                        "answer_relevance_prompts": {
                            "assessment": [
                                {
                                    "role": "user",
                                    "content": "You are presented with a response gener..."
                                }
                            ],
                            "positional_bias_assessment": [
                                {
                                    "role": "user",
                                    "content": "You are presented with a response gener..."
                                }
                            ],
                            "summarization": [
                                {
                                    "role": "user",
                                    "content": "Transform the following assessment into ..."
                                }
                            ],
                            "option_selection": [
                                {
                                    "content": "You are presented with a response gener...",
                                    "role": "user"
                                },
                                {
                                    "content": "To assess the quality of the response, l...",
                                    "role": "assistant"
                                },
                                {
                                    "content": "Now consider the evaluation criteria and...",
                                    "role": "user"
                                }
                            ],
                            "posional_bias_option_selection": [
                                {
                                    "content": "You are presented with a response gener...",
                                    "role": "user"
                                },
                                {
                                    "content": "To assess the quality of the response, l...",
                                    "role": "assistant"
                                },
                                {
                                    "content": "Now consider the evaluation criteria and...",
                                    "role": "user"
                                }
                            ]
                        },
                        "answer_relevance_option_selection_completion": "Could be Improved",
                        "answer_relevance_positional_bias_option_selection_completion": "Could be Improved",
                        "answer_relevance_criteria": "{    \"__type__\": \"criteria_with_options..."
                    }
                ]
        """
        logger.info(
            f'Starting evaluation with evaluator "{self.evaluator_name}" and provider "{self.inference_engine.get_pretty_print_name()}'
        )
        evaluations_count = len(predictions)
        # TODO: find out how to serialize and deserialize enums
        criterias = self.get_criteria(task_data, evaluations_count)
        self.__set_main_score(criterias)
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
            self.__get_parsed_criteria(criteria) for criteria in criterias
        ]

        (
            criteria_description_list,
            criteria_option_names_list,
            display_options_instruction_list,
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
        logger.info("The assessment was generated successfully.")

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
            logger.info("The summary was generated successfully.")

        option_selection_instances = [
            {
                "criteria_description": criteria_description,
                "display_options_instruction": display_options_instruction,
                "options": criteria_option_names,
                "data_classification_policy": ["public"],
            }
            for (
                criteria_description,
                display_options_instruction,
                criteria_option_names,
            ) in zip(
                criteria_description_list,
                display_options_instruction_list,
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
        logger.info("The selections were calculated successfully.")

        results = self.__get_results(
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
    """A judge for pairwise comparison evaluations, where two or more responses are compared to determine which one is preferred based on a criterion."""

    main_score = "1_winrate"
    """The main score metric for pairwise evaluation. By default, its value is `1_winrate`, and will take the value of the winrate of the first system."""
    reduction_map = {"mean": ["score"]}
    """A mapping specifying how scores should be reduced. By default, it will be ``{'main': ['score']}`` ."""

    def prepare(self):
        """Prepares the pairwise comparison by initializing the necessary templates and tasks. These tasks will be used to assess, summarize, and select options from candidate responses."""
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
        """Verifies that the criteria is of the correct type before processing the multi-stream data."""
        super().before_process_multi_stream()
        if self.criteria is not None and not isinstance(self.criteria, Criteria):
            raise Exception(
                f"The type of the criteria must be 'Criteria', instead it is of type '{type(self.criteria)}'"
            )
        return

    def __get_instance_results(
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
        criterion: Criteria,
    ):
        """Computes the results for each instance by comparing the responses and calculating metrics such as winrate, ranking, and the responses overall performance. This method processes assessment, summarization, and option selection outputs to track contest results, positional bias, and winrate.

        Args:
            instance_predictions (Dict[str, str]): The predictions for each response.
            assessment_prompts (List[str]): The prompts for the assessment task.
            assessment_outputs (List[str]): The results from the assessment task.
            summarization_prompts (List[str]): The prompts for the summarization task.
            summarization_outputs (List[str]): The results from the summarization task.
            option_selection_prompts (List[str]): The prompts for the option selection task.
            option_selection_outputs (List[str]): The results from the option selection task.
            selections (List[str]): The selections made during the pairwise comparison.
            contests_count (int): The total number of contests that were run.
            combination_indexes (List[Tuple[int, int]]): The indexes of the response pairs that were compared.
            criterion (Criteria): The criterion used to assess the responses.

        Returns:
            dict: A dictionary containing the results for each response, including winrate, ranking, and other metrics.
        """
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

        all_results["criteria"] = criterion.to_json()
        return self.clean_results(all_results)

    def __parse_prediction_to_dict(self, predictions: Union[Dict[str, str], List[str]]):
        """Converts a list or dictionary of predictions into a dictionary format.

        Args:
            predictions (Union[Dict[str, str], List[str]]): The prediction data to convert.

        Returns:
            dict: The prediction data in dictionary format.
        """
        if isinstance(predictions, list):
            return {f"{key + 1}": value for key, value in enumerate(predictions)}
        if isinstance(predictions, dict):
            return predictions
        raise UnitxtError(
            f"Prediction may be a list or a dict. Instead got type {type(predictions)}"
        )

    def __convert_predictions_to_dicts(
        self, predictions: Union[List[Dict[str, str]], List[str]]
    ):
        """Converts a list of predictions into a list of dictionaries.

        Args:
            predictions (Union[List[Dict[str, str]], List[str]]): The predictions to convert.

        Returns:
            List[dict]: A list of predictions in dictionary format.
        """
        return [
            self.__parse_prediction_to_dict(prediction) for prediction in predictions
        ]

    def __set_main_score(self, predictions: List[Dict[str, str]]):
        self.main_score = f"{next(iter(predictions[0].keys()))}_winrate"

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict[str, str]],
    ) -> List[Dict]:
        r"""Executes the pairwise comparison evaluation, including assessment, summarization, and option selection. It computes the winrate and ranking for the responses.

        Args:
            references (List[List[str]]): A list of reference responses for comparison.
            predictions (List[str]): A list of predicted responses.
            task_data (List[Dict[str, str]]): Task data to be used for evaluation.

        Returns:
        --------
        List[Dict[str,Dict]]
            The results of the evaluation, including winrate, ranking, and other metrics.

            For each instance result, the following metrics are included per response/system. Each of the metrics will have appended the systems name, if predictions were provided as a list of dicts, or their index, starting from 1, if predictions were provided as a list of lists.

            All the fields are arrays with length equal to `len(systems) - 1`. For any result at index `i`: `response_name[i]`'s contest against `compared_to[i]`'s result is `contest_results[i]`.

            Explanation of fields:

            - `summaries`: A list of LLM-generated summaries explaining the comparison results for each response.
            - `contest_results`: A list of boolean values indicating whether the response won in each comparison.
            - `selections`: A list of the selected system names, representing the preferred response in each comparison.
            - `compared_to`: A list of system names that were compared against the given response.
            - `assessments`: A list of LLM-generated assessments explaining the reasoning behind the evaluation results.
            - `positional_bias_assessments`: A list of LLM-generated assessments focused on detecting positional bias in the evaluation.
            - `option_selection_outputs`: A list of response names selected as the best choice based on the evaluation.
            - `positional_bias`: A list of boolean values indicating whether positional bias was detected in the contest.
            - `positional_bias_selection`: A list of response names representing the selected option when considering positional bias.
            - `prompts`: A dictionary of prompts used in different stages of evaluation.
                - `assessment`: The prompt used to instruct the model on how to assess the responses.
                - `positional_bias_assessment`: The prompt used to instruct the model on how to assess positional bias.
                - `option_selection`: The prompt used to guide the model in selecting the best response.
                - `positional_bias_option_selection`: The prompt used for selecting the best response while checking for positional bias.
                - `summary`: The prompt used to generate a summary of the assessment.
            - `winrate`: A float representing the proportion of comparisons the response won.
            - `llm_as_judge`: Equal to `winrate`.
            - `ranking`: An integer representing the ranking position of the response based on the evaluation results. Best is 1.
            - `response_name`: A string identifying the response in the evaluation.

            Result example:

            .. code-block:: python

                    [
                        {
                            "system1_contest_results": [
                                true,
                                true
                            ],
                            "system1_selections": [
                                "system1",
                                "system1"
                            ],
                            "system1_compared_to": [
                                "system2",
                                "system3"
                            ],
                            "system1_assessments": [
                                "To determine the better response accordi...",
                                "To determine the better response accordi..."
                            ],
                            "system1_positional_bias_assessments": [
                                "To determine the better response accordi...",
                                "To determine the better response accordi..."
                            ],
                            "system1_option_selection_outputs": [
                                "system1",
                                "system1"
                            ],
                            "system1_positional_bias": [
                                false,
                                false
                            ],
                            "system1_positional_bias_selection": [
                                "system1",
                                "system1"
                            ],
                            "system1_prompts": {
                                "assessment": [
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ],
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ]
                                ],
                                "positional_bias_assessment": [
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ],
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ]
                                ],
                                "option_selection": [
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ],
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ]
                                ],
                                "positional_bias_option_selection": [
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ],
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ]
                                ]
                            },
                            "system1_winrate": 1.0,
                            "system1_llm_as_judge": 1.0,
                            "system1_ranking": 1,
                            "system1_response_name": "system1",
                            "system2_contest_results": [
                                false,
                                true
                            ],
                            "system2_selections": [
                                "system1",
                                "system2"
                            ],
                            "system2_compared_to": [
                                "system1",
                                "system3"
                            ],
                            "system2_assessments": [
                                "To determine the better response accordi...",
                                "To determine the better response accordi..."
                            ],
                            "system2_positional_bias_assessments": [
                                "To determine the better response accordi...",
                                "To determine the better response accordi..."
                            ],
                            "system2_option_selection_outputs": [
                                "system1",
                                "system2"
                            ],
                            "system2_positional_bias": [
                                false,
                                false
                            ],
                            "system2_positional_bias_selection": [
                                "system1",
                                "system2"
                            ],
                            "system2_prompts": {
                                "assessment": [
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ],
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ]
                                ],
                                "positional_bias_assessment": [
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ],
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ]
                                ],
                                "option_selection": [
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ],
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ]
                                ],
                                "positional_bias_option_selection": [
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ],
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ]
                                ]
                            },
                            "system2_winrate": 0.5,
                            "system2_llm_as_judge": 0.5,
                            "system2_ranking": 2,
                            "system2_response_name": "system2",
                            "system3_contest_results": [
                                false,
                                false
                            ],
                            "system3_selections": [
                                "system1",
                                "system2"
                            ],
                            "system3_compared_to": [
                                "system1",
                                "system2"
                            ],
                            "system3_assessments": [
                                "To determine the better response accordi...",
                                "To determine the better response accordi..."
                            ],
                            "system3_positional_bias_assessments": [
                                "To determine the better response accordi...",
                                "To determine the better response accordi..."
                            ],
                            "system3_option_selection_outputs": [
                                "system1",
                                "system2"
                            ],
                            "system3_positional_bias": [
                                false,
                                false
                            ],
                            "system3_positional_bias_selection": [
                                "system1",
                                "system2"
                            ],
                            "system3_prompts": {
                                "assessment": [
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ],
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ]
                                ],
                                "positional_bias_assessment": [
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ],
                                    [
                                        {
                                            "role": "user",
                                            "content": "You are provided a pair of responses (Re..."
                                        }
                                    ]
                                ],
                                "option_selection": [
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ],
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ]
                                ],
                                "positional_bias_option_selection": [
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ],
                                    [
                                        {
                                            "content": "You are provided a pair of responses (Re...",
                                            "role": "user"
                                        },
                                        {
                                            "content": "To determine the better response accordi...",
                                            "role": "assistant"
                                        },
                                        {
                                            "content": "Now considering the evaluation criteria,...",
                                            "role": "user"
                                        }
                                    ]
                                ]
                            },
                            "system3_winrate": 0.0,
                            "system3_llm_as_judge": 0.0,
                            "system3_ranking": 3,
                            "system3_response_name": "system3",
                            "criteria": "{    \"__type__\": \"criteria\",    \"name\"..."
                        }
                    ]
        """
        logger.info(
            f'Starting evaluation with evaluator "{self.evaluator_name}" and provider {self.inference_engine.get_pretty_print_name()}'
        )
        predictions = self.__convert_predictions_to_dicts(predictions)
        self.__set_main_score(predictions)
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

        logger.info(
            f"The evaluation will perform {sum(contests_count_list) * [1, 2][self.check_positional_bias]} ({' + '.join([f'{c * [1, 2][self.check_positional_bias]}' for c in contests_count_list])}) pairwise comparisons"
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

        criterias = self.get_criteria(task_data, instances_count)
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
        logger.info("The assessment was generated successfully.")

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
            logger.info("The summary was generated successfully.")

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
        logger.info("The selections were calculated successfully.")
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
            instance_results = self.__get_instance_results(
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
