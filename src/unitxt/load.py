from .operator import StreamSource
from datasets import DatasetDict
from typing import Union

from .catalog import LocalCatalog
from .artifact import Artifact


def load_stream(source_name_or_path: str) -> StreamSource:
    if Artifact.is_artifact_file(source_name_or_path):
        return Artifact.load(source_name_or_path)
    else:
        return LocalCatalog().load(source_name_or_path)


def load_dataset(source: Union[StreamSource, str]) -> DatasetDict:
    assert isinstance(source, (StreamSource, str)), "source must be a StreamSource or a string"
    if isinstance(source, str):
        source = load_stream(source)
    return source().to_dataset()
