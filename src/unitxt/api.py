from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from datasets import DatasetDict

from .artifact import fetch_artifact
from .dataset_utils import get_dataset_artifact
from .logging_utils import get_logger
from .metric_utils import _compute, _inference_post_process
from .operator import SourceOperator
from .schema import UNITXT_DATASET_SCHEMA
from .standard import StandardRecipe

logger = get_logger()


def load(source: Union[SourceOperator, str]) -> DatasetDict:
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


def load_dataset(dataset_query: Optional[str] = None, **kwargs) -> DatasetDict:
    """Loads dataset.

    If the 'dataset_query' argument is provided, then dataset is loaded from a card in local
    catalog based on parameters specified in the query.
    Alternatively, dataset is loaded from a provided card based on explicitly given parameters.

    Args:
        dataset_query (str, optional): A string query which specifies a dataset to load from local catalog or name of specific recipe or benchmark in the catalog.
            For example:
            "card=cards.wnli,template=templates.classification.multi_class.relation.default".
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

    return recipe().to_dataset(features=UNITXT_DATASET_SCHEMA)


def evaluate(predictions, data) -> List[Dict[str, Any]]:
    return _compute(predictions=predictions, references=data)


def post_process(predictions, data) -> List[Dict[str, Any]]:
    return _inference_post_process(predictions=predictions, references=data)


@lru_cache
def _get_produce_with_cache(dataset_query: Optional[str] = None, **kwargs):
    return load_recipe(dataset_query, **kwargs).produce


def produce(instance_or_instances, dataset_query: Optional[str] = None, **kwargs):
    is_list = isinstance(instance_or_instances, list)
    if not is_list:
        instance_or_instances = [instance_or_instances]
    result = _get_produce_with_cache(dataset_query, **kwargs)(instance_or_instances)
    if not is_list:
        result = result[0]
    return result


def infer(
    instance_or_instances,
    engine,
    dataset_query: Optional[str] = None,
    return_data=False,
    **kwargs,
):
    dataset = produce(instance_or_instances, dataset_query, **kwargs)
    engine, _ = fetch_artifact(engine)
    raw_predictions = engine.infer(dataset)
    predictions = post_process(raw_predictions, dataset)
    if return_data:
        for prediction, raw_prediction, instance in zip(
            predictions, raw_predictions, dataset
        ):
            instance["prediction"] = prediction
            instance["raw_prediction"] = raw_prediction
        return dataset
    return predictions
