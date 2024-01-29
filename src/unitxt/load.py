from typing import Union

from datasets import DatasetDict

from .artifact import fetch_artifact
from .operator import SourceOperator


def load_dataset(source: Union[SourceOperator, str]) -> DatasetDict:
    assert isinstance(
        source, (SourceOperator, str)
    ), "source must be a SourceOperator or a string"
    if isinstance(source, str):
        source, _ = fetch_artifact(source)
    return source().to_dataset()
