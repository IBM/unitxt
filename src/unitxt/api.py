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
