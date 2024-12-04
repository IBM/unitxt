import json
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from .artifact import fetch_artifact
from .card import TaskCard
from .dataset_utils import get_dataset_artifact
from .inference import InferenceEngine, LogProbInferenceEngine
from .loaders import LoadFromDictionary
from .logging_utils import get_logger
from .metric_utils import _compute, _inference_post_process
from .metrics import Metric, MetricsList
from .operator import SourceOperator
from .schema import UNITXT_DATASET_SCHEMA, loads_instance
from .settings_utils import get_constants, get_settings
from .standard import StandardRecipe
from .task import Task
from .type_utils import isoftype

logger = get_logger()
constants = get_constants()
settings = get_settings()


def load(source: Union[SourceOperator, str]):
    assert isinstance(
        source, (SourceOperator, str)
    ), "source must be a SourceOperator or a string"
    if isinstance(source, str):
        source, _ = fetch_artifact(source)
    return source().to_dataset()


def _get_recipe_from_query(dataset_query: str) -> StandardRecipe:
    dataset_query = dataset_query.replace("sys_prompt", "instruction")
    try:
        dataset_stream, _ = fetch_artifact(dataset_query)
    except:
        dataset_stream = get_dataset_artifact(dataset_query)
    return dataset_stream


def _get_recipe_from_dict(dataset_params: Dict[str, Any]) -> StandardRecipe:
    recipe_attributes = list(StandardRecipe.__dict__["__fields__"].keys())
    for param in dataset_params.keys():
        assert param in recipe_attributes, (
            f"The parameter '{param}' is not an attribute of the 'StandardRecipe' class. "
            f"Please check if the name is correct. The available attributes are: '{recipe_attributes}'."
        )
    return StandardRecipe(**dataset_params)


def _verify_dataset_args(dataset_query: Optional[str] = None, dataset_args=None):
    if dataset_query and dataset_args:
        raise ValueError(
            "Cannot provide 'dataset_query' and key-worded arguments at the same time. "
            "If you want to load dataset from a card in local catalog, use query only. "
            "Otherwise, use key-worded arguments only to specify properties of dataset."
        )

    if dataset_query:
        if not isinstance(dataset_query, str):
            raise ValueError(
                f"If specified, 'dataset_query' must be a string, however, "
                f"'{dataset_query}' was provided instead, which is of type "
                f"'{type(dataset_query)}'."
            )

    if not dataset_query and not dataset_args:
        raise ValueError(
            "Either 'dataset_query' or key-worded arguments must be provided."
        )


def load_recipe(dataset_query: Optional[str] = None, **kwargs) -> StandardRecipe:
    if isinstance(dataset_query, StandardRecipe):
        return dataset_query

    _verify_dataset_args(dataset_query, kwargs)

    if dataset_query:
        recipe = _get_recipe_from_query(dataset_query)

    if kwargs:
        recipe = _get_recipe_from_dict(kwargs)

    return recipe


def create_dataset(
    task: str,
    test_set: List[Dict[Any, Any]],
    train_set: Optional[List[Dict[Any, Any]]] = None,
    validation_set: Optional[List[Dict[Any, Any]]] = None,
    **kwargs,
):
    """Creates dataset from input data based on a specific task.

    Args:
        task:  The name of the task from the Unitxt Catalog (https://www.unitxt.ai/en/latest/catalog/catalog.tasks.__dir__.html)
        test_set : required list of instances
        train_set : optional train_set
        validation_set: optional validation set
        **kwargs: Arguments used to load dataset from provided datasets (see load_dataset())

    Returns:
        DatasetDict

    Example:
        template = Template(...)
        dataset = create_dataset(task="tasks.qa.open", template=template, format="formats.chatapi")
    """
    data = {"test": test_set}
    if train_set is not None:
        data["train"] = train_set
    if validation_set is not None:
        data["validation"] = validation_set
    card = TaskCard(loader=LoadFromDictionary(data=data), task=task)
    return load_dataset(card=card, **kwargs)


def load_dataset(
    dataset_query: Optional[str] = None,
    split: Optional[str] = None,
    streaming: bool = False,
    disable_cache: Optional[bool] = None,
    **kwargs,
) -> Union[DatasetDict, IterableDatasetDict, Dataset, IterableDataset]:
    """Loads dataset.

    If the 'dataset_query' argument is provided, then dataset is loaded from a card in local
    catalog based on parameters specified in the query.
    Alternatively, dataset is loaded from a provided card based on explicitly given parameters.

    Args:
        dataset_query (str, optional): A string query which specifies a dataset to load from local catalog or name of specific recipe or benchmark in the catalog.
            For example:
            "card=cards.wnli,template=templates.classification.multi_class.relation.default".
        streaming (bool, False): When True yields the data as Unitxt streams dictionary
        split (str, optional): The split of the data to load
        disable_cache (str, optional): Disable caching process of the data
        **kwargs: Arguments used to load dataset from provided card, which is not present in local catalog.

    Returns:
        DatasetDict

    Examples:
        dataset = load_dataset(
            dataset_query="card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5"
        )  # card must be present in local catalog

        card = TaskCard(...)
        template = Template(...)
        loader_limit = 10
        dataset = load_dataset(card=card, template=template, loader_limit=loader_limit)
    """
    recipe = load_recipe(dataset_query, **kwargs)

    stream = recipe()
    if split is not None:
        stream = stream[split]

    if disable_cache is None:
        disable_cache = settings.disable_hf_datasets_cache

    if streaming:
        return stream.to_iterable_dataset(
            features=UNITXT_DATASET_SCHEMA,
        ).map(loads_instance, batched=True)

    return stream.to_dataset(
        features=UNITXT_DATASET_SCHEMA, disable_cache=disable_cache
    ).with_transform(loads_instance)


def evaluate(predictions, data):
    return _compute(predictions=predictions, references=data)


def create_and_evaluate_dataset(
    predictions: List[Any],
    data: List[Dict[str, Any]],
    task: Task,
    metrics: Optional[List[Union[Metric, MetricsList]]] = None,
) -> List[Dict[str, Any]]:
    """Creates and evaluates dataset from input data based on a specific Unitxt task.

    Args:
        predictions: The predictions from the model which are the input to evaluation.
        The type of expected predictions is defined in the task's 'prediction_type' field.
        task:  The name of the task from the Unitxt Catalog (https://www.unitxt.ai/en/latest/catalog/catalog.tasks.__dir__.html)
        data : Required list of instances to evaluate.
        metrics : Optional list of metrics to use.  If not specified, the default metrics defined in the task are used.

    Returns:
        output dataset with evaluated scores (see https://www.unitxt.ai/en/latest/docs/evaluating_datasets.html)

    Example:
        dataset = create_and_evaluate_dataset(task="tasks.qa.open", data)
    """
    task, _ = fetch_artifact(task)

    if task.default_template is None:
        raise Exception(
            f"create_and_evaluate_dataset requires the given task ('{task.__id__}') to template set in the 'default_template' field "
        )
    # We assume the user provides predictions in the required format for the task.

    for prediction in predictions:
        if not isoftype(prediction, task.prediction_type):
            raise Exception(
                f"The prediction passed to 'create_and_evaluate_dataset' does not match the task's required prediction type of '{task.prediction_type}'. The prediction is : {prediction}"
            )
        task.default_template.postprocessors = []
    data = create_dataset(
        task=task, test_set=data, split="test", format="formats.empty", metrics=metrics
    )
    return _compute(predictions=predictions, references=data)


def post_process(predictions, data) -> List[Dict[str, Any]]:
    return _inference_post_process(predictions=predictions, references=data)


@lru_cache
def _get_produce_with_cache(dataset_query: Optional[str] = None, **kwargs):
    return load_recipe(dataset_query, **kwargs).produce


def produce(
    instance_or_instances, dataset_query: Optional[str] = None, **kwargs
) -> Union[Dataset, Dict[str, Any]]:
    is_list = isinstance(instance_or_instances, list)
    if not is_list:
        instance_or_instances = [instance_or_instances]
    result = _get_produce_with_cache(dataset_query, **kwargs)(instance_or_instances)
    if not is_list:
        return result[0]
    return Dataset.from_list(result).with_transform(loads_instance)


def infer(
    instance_or_instances,
    engine: InferenceEngine,
    dataset_query: Optional[str] = None,
    return_data: bool = False,
    return_log_probs: bool = False,
    return_meta_data: bool = False,
    **kwargs,
):
    dataset = produce(instance_or_instances, dataset_query, **kwargs)
    engine, _ = fetch_artifact(engine)
    if return_log_probs:
        if not isinstance(engine, LogProbInferenceEngine):
            raise NotImplementedError(
                f"Error in infer: return_log_probs set to True but supplied engine "
                f"{engine.__class__.__name__} does not support logprobs."
            )
        infer_outputs = engine.infer_log_probs(dataset, return_meta_data)
        raw_predictions = (
            [output.prediction for output in infer_outputs]
            if return_meta_data
            else infer_outputs
        )
        raw_predictions = [
            json.dumps(raw_prediction) for raw_prediction in raw_predictions
        ]
    else:
        infer_outputs = engine.infer(dataset, return_meta_data)
        raw_predictions = (
            [output.prediction for output in infer_outputs]
            if return_meta_data
            else infer_outputs
        )
    predictions = post_process(raw_predictions, dataset)
    if return_data:
        if return_meta_data:
            infer_output_list = [
                infer_output.__dict__ for infer_output in infer_outputs
            ]
            for infer_output in infer_output_list:
                del infer_output["prediction"]
            dataset = dataset.add_column("infer_meta_data", infer_output_list)
        dataset = dataset.add_column("prediction", predictions)
        return dataset.add_column("raw_prediction", raw_predictions)
    return predictions
