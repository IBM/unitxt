import itertools
from abc import abstractmethod
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
from .metric_utils import EmptyPrediction
from .metrics import MapReduceMetric
from .task import Task
from .templates import Template

logger = get_logger(__name__)


class LLMJudge(MapReduceMetric[Any, Dict[str, Any]]):
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
    """Fields to be used as context. If a dict is provided, the keys are used as the final names in the prompts, while the values are used to access the context variable values in the `task_data` object (it is recommended to provide the context_fields in the Criteria `context_fields` field as this field will be deprecated in the future)."""

    generate_summaries: bool = False
    """Flag to generate summaries of the assessments. Defaults to `False`."""

    format: str = "formats.chat_api"
    """The format used for the inference. Defaults to `formats.chat_api` (only allowed value)."""

    include_prompts_in_result: bool = True
    """Flag to include prompts in the result. Defaults to `True`."""

    criteria_field: str = None
    """The field specifying the evaluation criteria in the `task_data` object. If the `criteria` is provided, it will take precedence."""

    criteria: Criteria = None
    """The criteria used for evaluation."""

    def prepare(self):
        """Prepares the `LLMJudge` instance by setting up context fields and evaluator name."""
        super().prepare()
        self.context_fields = self.get_context_fields_as_dict(self.context_fields)

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

    def get_context_fields_as_dict(self, context_fields: Union[str, List, Dict]):
        result = context_fields if context_fields else {}
        if isinstance(result, str):
            result = [result]
        if isinstance(result, List):
            result = {context_field: context_field for context_field in result}
        return result

    def get_contexts(
        self, task_data: List[Dict[str, Any]], criteria: List[Criteria]
    ) -> List[Dict[str, str]]:
        """Extracts and parses context fields from task data.

        Args:
            task_data (List[Dict[str, Any]]): The task data containing context information.
            criteria ( List[Criteria]): The criteria list from which to take the context fields if they weren't provided in the self.context_fields field

        Returns:
            List[Dict[str, str]]: A list of parsed context dictionaries.
        """
        parsed_contexts = []
        for i, td in enumerate(task_data):
            context_fields_for_td = self.context_fields
            if not context_fields_for_td and criteria[i].context_fields:
                context_fields_for_td = self.get_context_fields_as_dict(
                    criteria[i].context_fields
                )

            parsed_contexts.append(
                get_parsed_context(
                    {
                        context_field_name: dict_get(td, context_field)
                        for context_field_name, context_field in context_fields_for_td.items()
                    }
                )
            )
        return parsed_contexts

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

    def get_criteria(self, task_data, eval_count) -> List[Criteria]:
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
            criteria_list = [
                fetch_artifact(task_data_instance[self.criteria_field])[0]
                for task_data_instance in task_data
            ]
        else:
            logger.info(
                "Reading criteria from self. Criteria is a single CriteriaWithOptions, replicating it for all predictions"
            )
            criteria_list: List[Criteria] = [self.criteria] * eval_count
        unique_criteria_names = list({criteria.name for criteria in criteria_list})

        logger.info(f"Criteria names are '{', '.join(unique_criteria_names)}'")
        return criteria_list

    def get_predictions(
        self,
        task_data: List[Dict[str, Any]],
        criteria: List[Criteria],
        predictions: List[str],
    ) -> List[str]:
        if not predictions or all(
            (
                isinstance(prediction, EmptyPrediction)
                or prediction == str(EmptyPrediction())
            )
            for prediction in predictions
        ):
            predictions_from_task_data = []
            for i, td in enumerate(task_data):
                if (
                    criteria[i].prediction_field is not None
                    and criteria[i].prediction_field in td
                ):
                    predictions_from_task_data.append(
                        dict_get(td, criteria[i].prediction_field)
                    )
                else:
                    raise UnitxtError(
                        "You must set either the predictions in the evaluate() call or specify the prediction field name to be taken from the task_data using the `Criteria`'s prediction_field field."
                    )
            return predictions_from_task_data

        return predictions

    def map(
        self, 
        prediction: Any, 
        references: List[Any], 
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single instance using the LLM judge.
        
        This method processes one instance at a time, but the actual implementation
        will collect instances and process them in batches for efficiency.
        """
        # This method is implemented by subclasses like LLMJudgeDirect
        raise NotImplementedError("Subclasses must implement map method")

    def map_stream(self, evaluation_inputs_stream):
        """Common map_stream implementation for all LLM judge subclasses."""
        logger.info(
            f'Starting evaluation with {self.__class__.__name__} using "{self.inference_engine.get_engine_id()}"'
        )

        # Prepare all instances for inference without aggregating heavy data
        prepared_instances = []
        for prediction, references, task_data in evaluation_inputs_stream:
            prepared_instance = self._prepare_instance_for_inference(prediction, references, task_data)
            prepared_instances.append(prepared_instance)
        
        # Optional: Set main score for judges that need it (like LLMJudgeDirect)
        if prepared_instances and hasattr(self, '_LLMJudgeDirect__set_main_score'):
            self._LLMJudgeDirect__set_main_score([prepared_instances[0]['criteria']])
        
        # Run all inference steps on the prepared instances
        return self._run_inference_on_all(prepared_instances)
    
    @abstractmethod
    def _prepare_instance_for_inference(self, prediction, references, task_data):
        """Prepare a single instance for inference without keeping heavy data.
        
        This method should be implemented by each judge subclass to prepare
        an individual instance for batch inference processing.
        """
        pass
    
    @abstractmethod  
    def _run_inference_on_all(self, prepared_instances):
        """Run inference on all prepared instances efficiently.
        
        This method should be implemented by each judge subclass to execute
        inference on the batch of prepared instances and return results.
        """
        pass

    def reduce(self, intermediates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual instance results into global scores."""
        if not intermediates:
            return {}
        
        # Collect all numeric scores for averaging
        numeric_scores = {}
        for result in intermediates:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_scores:
                        numeric_scores[key] = []
                    numeric_scores[key].append(value)
        
        # Calculate averages
        aggregated = {}
        for key, values in numeric_scores.items():
            if values:
                aggregated[key] = sum(values) / len(values)
        
        # Set the main score
        if self.main_score in aggregated:
            aggregated["score"] = aggregated[self.main_score]
            aggregated["score_name"] = self.main_score
        
        return aggregated

    def reduce_one(self, intermediate: Dict[str, Any]) -> Dict[str, Any]:
        """Return individual instance scores."""
        result = dict(intermediate)
        if self.main_score in result:
            result["score"] = result[self.main_score]
            result["score_name"] = self.main_score
        return result


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

    def map(
        self, 
        prediction: Any, 
        references: List[Any], 
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "LLMJudgeDirect uses map_stream for efficient batch processing, not map"
        )


    def _prepare_instance_for_inference(self, prediction, references, task_data):
        """Prepare a single instance for inference without keeping heavy data."""
        # Get criteria for this specific instance
        criteria = self.get_criteria([task_data], 1)[0]
        
        # Get prediction for this specific instance
        pred = self.get_predictions([task_data], [criteria], [prediction])[0]
        
        # Get context for this specific instance
        context = self.get_contexts([task_data], [criteria])[0]
        
        # Parse criteria for this instance
        criteria_description, criteria_option_names, display_options_instruction = self.__get_parsed_criteria(criteria)
        
        return {
            'prediction': pred,
            'context': context,
            'criteria': criteria,
            'criteria_description': criteria_description,
            'criteria_option_names': criteria_option_names,
            'display_options_instruction': display_options_instruction,
        }

    def _run_inference_on_all(self, prepared_instances):
        """Run inference on all prepared instances efficiently."""
        # Prepare all assessment instances
        assessment_instances = []
        instance_metadata = []
        
        for i, prep in enumerate(prepared_instances):
            # Store metadata for later use
            instance_metadata.append({
                'criteria': prep['criteria'],
                'criteria_description': prep['criteria_description'],
                'criteria_option_names': prep['criteria_option_names'],
                'display_options_instruction': prep['display_options_instruction'],
                'original_index': i
            })
            
            # Create assessment instance
            assessment_instances.append({
                "context_variables": prep['context'],
                "response": prep['prediction'],
                "display_options_instruction": prep['display_options_instruction'],
                "criteria_description": prep['criteria_description'],
                "data_classification_policy": ["public"],
            })
            
            # If checking positional bias, add reversed version
            if self.check_positional_bias:
                reversed_criteria = CriteriaWithOptions(
                    name=prep['criteria'].name,
                    description=prep['criteria'].description,
                    option_map=prep['criteria'].option_map,
                    options=list(reversed(prep['criteria'].options)),
                )
                rev_criteria_description, rev_criteria_option_names, rev_display_options_instruction = self.__get_parsed_criteria(reversed_criteria)
                
                # Store reversed metadata
                instance_metadata.append({
                    'criteria': reversed_criteria,
                    'criteria_description': rev_criteria_description,
                    'criteria_option_names': rev_criteria_option_names,
                    'display_options_instruction': rev_display_options_instruction,
                    'original_index': i,
                    'is_positional_bias': True
                })
                
                # Add reversed assessment instance
                assessment_instances.append({
                    "context_variables": prep['context'],
                    "response": prep['prediction'],
                    "display_options_instruction": rev_display_options_instruction,
                    "criteria_description": rev_criteria_description,
                    "data_classification_policy": ["public"],
                })
        
        # Perform assessment step on all instances at once
        assessment_prompts, assessment_outputs, _ = self.perform_evaluation_step(
            assessment_instances, self.assessment_task, self.assessment_template
        )
        logger.info("The assessment was generated successfully.")
        
        # Summarization step (if enabled)
        summarization_prompts = None
        summarization_outputs = None
        if self.generate_summaries:
            evaluations_count = len(prepared_instances)
            summarization_instances = [
                {
                    "assessment": assessment_output,
                    "data_classification_policy": ["public"],
                }
                for assessment_output in assessment_outputs[:evaluations_count]  # Only original assessments, not positional bias
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
        
        # Option selection step
        option_selection_instances = []
        for metadata in instance_metadata:
            option_selection_instances.append({
                "criteria_description": metadata['criteria_description'],
                "display_options_instruction": metadata['display_options_instruction'],
                "options": metadata['criteria_option_names'],
                "data_classification_policy": ["public"],
            })
        
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
        
        # Process results for each original instance
        evaluations_count = len(prepared_instances)
        criteria_list = [meta['criteria'] for meta in instance_metadata]
        
        results = self.__get_results(
            assessment_prompts,
            assessment_outputs,
            summarization_prompts,
            summarization_outputs,
            option_selection_prompts,
            option_selection_outputs,
            selections,
            evaluations_count,
            criteria_list,
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

    def map(
        self, 
        prediction: Any, 
        references: List[Any], 
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "LLMJudgePairwise uses map_stream for efficient batch processing, not map"
        )


    def _prepare_instance_for_inference(self, prediction, references, task_data):
        """Prepare a single instance for pairwise inference without keeping heavy data."""
        # Get criteria for this specific instance
        criteria = self.get_criteria([task_data], 1)[0]
        
        # Get context for this specific instance
        context = self.get_contexts([task_data], [criteria])[0]
        
        # Get predictions for this specific instance and convert to dicts
        pred = self.get_predictions([task_data], [criteria], [prediction])[0]
        pred_dict = self.__parse_prediction_to_dict(pred)
        
        return {
            'prediction_dict': pred_dict,
            'context': context,
            'criteria': criteria,
            'task_data': task_data,
        }

    def _run_inference_on_all(self, prepared_instances):
        """Run pairwise inference on all prepared instances efficiently."""
        if not prepared_instances:
            return []
        
        # Set main score and reduction map based on first instance
        first_pred_dict = prepared_instances[0]['prediction_dict']
        self.__set_main_score([first_pred_dict])
        self.reduction_map = {"mean": ["score"]}
        self.reduction_map["mean"].extend(
            [f"{key}_winrate" for key in first_pred_dict.keys()]
        )
        
        # Prepare all assessment instances without keeping heavy data aggregated
        all_assessment_instances = []
        all_instance_metadata = []
        
        for prep_instance in prepared_instances:
            assessment_instances, instance_metadata = self._prepare_assessment_instances(prep_instance)
            all_assessment_instances.extend(assessment_instances)
            all_instance_metadata.append(instance_metadata)
        
        # Perform assessment step on all instances at once
        assessment_prompts, assessment_outputs, _ = self.perform_evaluation_step(
            all_assessment_instances, self.assessment_task, self.assessment_template
        )
        logger.info("The assessment was generated successfully.")
        
        # Summarization step (if enabled)
        summarization_prompts = None
        summarization_outputs = None
        if self.generate_summaries:
            summarization_instances = []
            for metadata in all_instance_metadata:
                for i in range(metadata['contests_count']):
                    summarization_instances.append({
                        "assessment": assessment_outputs[metadata['assessment_start_idx'] + i],
                        "data_classification_policy": ["public"],
                    })
            
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
        
        # Option selection step
        option_selection_instances = []
        for assessment_instance in all_assessment_instances:
            option_selection_instances.append({
                "options": [f"Response {option}" for option in assessment_instance["option_pair"]],
                "score_option_instruction": assessment_instance["score_option_instruction"],
                "data_classification_policy": ["public"],
            })
        
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
        
        # Process results for each instance
        results = []
        assessment_idx = 0
        for i, (prep_instance, metadata) in enumerate(zip(prepared_instances, all_instance_metadata)):
            contests_count = metadata['contests_count']
            combination_indexes = metadata['combination_indexes']
            
            slice_end = assessment_idx + contests_count
            if self.check_positional_bias:
                slice_end += contests_count
            
            sli = slice(assessment_idx, slice_end)
            sli_summarization = slice(assessment_idx, assessment_idx + contests_count) if self.generate_summaries else None
            
            instance_results = self.__get_instance_results(
                prep_instance['prediction_dict'],
                assessment_prompts[sli],
                assessment_outputs[sli],
                summarization_prompts[sli_summarization] if self.generate_summaries else None,
                summarization_outputs[sli_summarization] if self.generate_summaries else None,
                option_selection_prompts[sli],
                option_selection_outputs[sli],
                selections[sli],
                contests_count,
                combination_indexes,
                prep_instance['criteria'],
            )
            results.append(instance_results)
            assessment_idx = slice_end
        
        return results

    def _prepare_assessment_instances(self, prep_instance):
        """Prepare assessment instances for a single prepared instance."""
        pred_dict = prep_instance['prediction_dict']
        context = prep_instance['context']
        criteria = prep_instance['criteria']
        
        # Calculate combinations and prepare pairs
        prediction_names = list(pred_dict.keys())
        combination_indexes = list(itertools.combinations(range(len(prediction_names)), 2))
        contests_count = len(combination_indexes)
        
        # Prepare response pairs and option pairs
        response_pairs = []
        option_pairs = []
        for combination in combination_indexes:
            (idx_1, idx_2) = combination
            response_name_1 = prediction_names[idx_1]
            response_name_2 = prediction_names[idx_2]
            response_pairs.append([pred_dict[response_name_1], pred_dict[response_name_2]])
            option_pairs.append([response_name_1, response_name_2])
        
        # If checking positional bias, add reversed pairs
        if self.check_positional_bias:
            response_pairs += [list(reversed(pair)) for pair in response_pairs]
            option_pairs += [list(reversed(pair)) for pair in option_pairs]
        
        # Create assessment instances
        assessment_instances = []
        for response_pair, option_pair in zip(response_pairs, option_pairs):
            score_option_instruction = "".join([
                f'Choose "{option}" if Response {option} is better quality.\n'
                for option in option_pair
            ])
            
            assessment_instances.append({
                "context_variables": context,
                "response_a": response_pair[0],
                "response_b": response_pair[1],
                "option_a": option_pair[0],
                "option_b": option_pair[1],
                "criteria_name": criteria.name,
                "criteria_description": criteria.description,
                "data_classification_policy": ["public"],
                "option_pair": option_pair,  # Store for later use
                "score_option_instruction": score_option_instruction,
            })
        
        metadata = {
            'contests_count': contests_count,
            'combination_indexes': combination_indexes,
            'assessment_start_idx': 0,  # Will be set correctly when called
        }
        
        return assessment_instances, metadata
