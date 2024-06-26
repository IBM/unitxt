from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from datasets import DatasetDict

from .artifact import fetch_artifact
from .dataset_utils import get_dataset_artifact
from .logging_utils import get_logger
from .metric_utils import _compute, _post_process
from .operator import SourceOperator
from .standard import StandardRecipe

logger = get_logger()


def load(source: Union[SourceOperator, str]) -> DatasetDict:
    assert isinstance(
        source, (SourceOperator, str)
    ), "source must be a SourceOperator or a string"
    if isinstance(source, str):
        source, _ = fetch_artifact(source)
    return source().to_dataset()


def _load_dataset_from_query(dataset_query: str) -> DatasetDict:
    dataset_query = dataset_query.replace("sys_prompt", "instruction")
    dataset_stream = get_dataset_artifact(dataset_query)
    return dataset_stream().to_dataset()


def _load_dataset_from_dict(dataset_params: Dict[str, Any]) -> DatasetDict:
    recipe_attributes = list(StandardRecipe.__dict__["__fields__"].keys())
    for param in dataset_params.keys():
        assert param in recipe_attributes, (
            f"The parameter '{param}' is not an attribute of the 'StandardRecipe' class. "
            f"Please check if the name is correct. The available attributes are: '{recipe_attributes}'."
        )
    recipe = StandardRecipe(**dataset_params)
    return recipe().to_dataset()


def load_dataset(dataset_query: Optional[str] = None, **kwargs) -> DatasetDict:
    """Loads dataset.

    If the 'dataset_query' argument is provided, then dataset is loaded from a card in local
    catalog based on parameters specified in the query.
    Alternatively, dataset is loaded from a provided card based on explicitly given parameters.

    Args:
        dataset_query (str, optional): A string query which specifies dataset to load from local catalog.
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
    if dataset_query and kwargs:
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
        return _load_dataset_from_query(dataset_query)

    if kwargs:
        return _load_dataset_from_dict(kwargs)

    raise ValueError("Either 'dataset_query' or key-worded arguments must be provided.")


def evaluate(predictions, data) -> List[Dict[str, Any]]:
    return _compute(predictions=predictions, references=data)


def post_process(predictions, data) -> List[Dict[str, Any]]:
    return _post_process(predictions=predictions, references=data)


@lru_cache
def _get_produce_with_cache(recipe_query):
    return get_dataset_artifact(recipe_query).produce


def produce(instance_or_instances, recipe_query):
    is_list = isinstance(instance_or_instances, list)
    if not is_list:
        instance_or_instances = [instance_or_instances]
    result = _get_produce_with_cache(recipe_query)(instance_or_instances)
    if not is_list:
        result = result[0]
    return result


def infer(instance_or_instances, recipe, engine):
    dataset = produce(instance_or_instances, recipe)
    engine, _ = fetch_artifact(engine)
    predictions = engine.infer(dataset)
    return post_process(predictions, dataset)
