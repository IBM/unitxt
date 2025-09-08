from difflib import get_close_matches
from typing import Any, Dict, List, Union, cast

from .artifact import fetch_artifact
from .dict_utils import dict_get
from .error_utils import UnitxtError
from .inference import (
    InferenceEngine,
    PackageRequirementsMixin,
)
from .llm_as_judge_constants import (
    Criteria,
    CriteriaWithOptions,
)
from .logging_utils import get_logger
from .metric_utils import EmptyPrediction
from .metrics import BulkInstanceMetric

logger = get_logger(__name__)


class EvalAssistLLMJudge(BulkInstanceMetric, PackageRequirementsMixin):
    """A metric class to evaluate instances using LLM as a Judge.

    Evaluations are performed in two steps. First, the LLM is asked to generate an assessment following a CoT approach based on the criteria. Then, the same LLM is asked to select one of the available options. A summary of the general assessment can be generated for easy consumption by end users.
    """

    _requirements_list = {
        "evalassist": "Install huggingface package using 'pip install --upgrade evalassist",
    }

    inference_engine: InferenceEngine
    """The engine used for generating predictions in the different evaluation steps."""

    context_fields: Union[str, List[str], Dict[str, str], None] = None
    """Fields to be used as context. If a dict is provided, the keys are used as the final names in the prompts, while the values are used to access the context variable values in the `task_data` object (it is recommended to provide the context_fields in the Criteria `context_fields` field as this field will be deprecated in the future)."""

    check_positional_bias: bool = False
    """Flag to check for positional bias. Detecting for positional bias duplicates the amount of inference calls."""

    criteria_field: Union[str, None] = None
    """The field specifying the evaluation criteria in the `task_data` object. If the `criteria` is provided, it will take precedence."""

    criteria: Criteria = None
    """The criteria used for evaluation."""

    def prepare(self):
        """Prepares the `LLMJudge` instance by setting up context fields and evaluator name."""
        if self.context_fields is not None:
            self.context_fields = self.get_context_fields_as_dict(self.context_fields)
        super().prepare()

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

    def get_context_fields_as_dict(
        self, context_fields: Union[str, List[str], Dict[str, str]]
    ):
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
                {
                    context_field_name: str(dict_get(td, context_field))
                    for context_field_name, context_field in context_fields_for_td.items()
                }
            )
        return parsed_contexts

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
        criteria_list: List[Criteria]
        if self.criteria is None:
            if any(self.criteria_field not in td for td in task_data):
                raise UnitxtError(
                    f"The criteria field `{self.criteria_field}` required for {__class__.__name__} is not found in instance. Perhaps you meant '{get_close_matches(cast(str, self.criteria_field), task_data[0].keys(), n=1, cutoff=0.0)[0]}'?"
                )
            logger.info(
                f"Reading criteria from the task_data field '{self.criteria_field}'"
            )
            criteria_list = [
                cast(
                    Criteria, fetch_artifact(task_data_instance[self.criteria_field])[0]
                )
                for task_data_instance in task_data
            ]
        else:
            logger.info(
                "Reading criteria from self. Criteria is a single CriteriaWithOptions, replicating it for all predictions"
            )
            criteria_list = [self.criteria] * eval_count
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
                        dict_get(td, criteria[i].prediction_field)  # type: ignore
                    )
                else:
                    raise UnitxtError(
                        "You must set either the predictions in the evaluate() call or specify the prediction field name to be taken from the task_data using the `Criteria`'s prediction_field field."
                    )
            return predictions_from_task_data

        return predictions


class EvalAssistLLMJudgeDirect(EvalAssistLLMJudge):
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

    def __set_main_score(self, criterias: List[Criteria]):
        unique_criteria_names = list({criteria.name for criteria in criterias})
        if len(unique_criteria_names) == 1 and criterias[0].name != "":
            self.main_score = "_".join(criterias[0].name.lower().split(" "))
            self.reduction_map = {"mean": [self.main_score]}

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict[str, Any]],
    ) -> List[Dict]:
        logger.info(
            f"Starting evaluation with judge {self.inference_engine.get_pretty_print_name()}"
        )
        from evalassist.judges import (
            Criteria,
            DirectInstance,
            DirectInstanceResult,
            DirectJudge,
        )

        judge = DirectJudge(self.inference_engine)

        evaluations_count = len(task_data)
        # TODO: find out how to serialize and deserialize enums
        criteria = self.get_criteria(task_data, evaluations_count)
        self.__set_main_score(criteria)
        predictions = self.get_predictions(task_data, criteria, predictions)
        contexts = self.get_contexts(task_data, criteria)
        eval_assist_criteria = [
            Criteria.from_unitxt_criteria(criterion) for criterion in criteria
        ]

        instances = [
            DirectInstance(
                context=context,
                response=prediction,
            )
            for prediction, context in zip(predictions, contexts)
        ]

        results: list[DirectInstanceResult] = judge(
            instances=instances,
            criteria=eval_assist_criteria,
            check_positional_bias=self.check_positional_bias,
        )

        parsed_results: list[dict] = [
            {
                "selected_option": r.option,
                "explanation": r.explanation,
                "feedback": r.feedback if r.feedback is not None else None,
                "prompt": r.metadata["prompt"],
                "positional_bias": r.positional_bias.detected
                if r.positional_bias is not None
                else None,
                self.main_score: r.score if r.score is not None else 0.0,
            }
            for r in results
        ]

        parsed_results = [
            {
                f"{self.main_score}_{k}" if k != self.main_score else self.main_score: v
                for k, v in r.items()
            }
            for r in parsed_results
        ]

        return self.clean_results(parsed_results)
