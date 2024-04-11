from functools import lru_cache
from typing import Any, Dict, List, Union

from datasets import DatasetDict

from .artifact import fetch_artifact
from .dataset_utils import get_dataset_artifact
from .logging_utils import get_logger
from .metric_utils import _compute
from .operator import SourceOperator

logger = get_logger()


def load(source: Union[SourceOperator, str]) -> DatasetDict:
    assert isinstance(
        source, (SourceOperator, str)
    ), "source must be a SourceOperator or a string"
    if isinstance(source, str):
        source, _ = fetch_artifact(source)
    return source().to_dataset()


def load_dataset(dataset_query: str) -> DatasetDict:
    dataset_query = dataset_query.replace("sys_prompt", "instruction")
    dataset_stream = get_dataset_artifact(dataset_query)
    return dataset_stream().to_dataset()


def evaluate(predictions, data) -> List[Dict[str, Any]]:
    return _compute(predictions=predictions, references=data)


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
