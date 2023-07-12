from typing import Mapping, Optional, Sequence, Union

from datasets import load_dataset as hf_load_dataset

from .operator import SourceOperator
from .stream import MultiStream


class Loader(SourceOperator):
    pass


class LoadHF(Loader):
    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None
    cached = False

    def process(self):
        dataset = hf_load_dataset(
            self.path, name=self.name, data_dir=self.data_dir, data_files=self.data_files, streaming=True
        )

        return MultiStream.from_iterables(dataset)
