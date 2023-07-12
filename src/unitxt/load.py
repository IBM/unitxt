from typing import Union
from datasets import DatasetDict
from .artifact import fetch_artifact
from .operator import StreamSource


def load_dataset(source: Union[StreamSource, str]) -> DatasetDict:
    assert isinstance(source, (StreamSource, str)), "source must be a StreamSource or a string"
    if isinstance(source, str):
        artifact, _ = fetch_artifact(source)
    return artifact().to_dataset()
