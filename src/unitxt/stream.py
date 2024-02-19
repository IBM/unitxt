import tempfile
from typing import Dict, Iterable

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from .dataclass import Dataclass, OptionalField
from .generator_utils import CopyingReusableGenerator, ReusableGenerator


class Stream(Dataclass):
    """A class for handling streaming data in a customizable way.

    This class provides methods for generating, caching, and manipulating streaming data.

    Attributes:
        generator (function): A generator function for streaming data. :no-index:
        gen_kwargs (dict, optional): A dictionary of keyword arguments for the generator function. :no-index:
        caching (bool): Whether the data is cached or not. :no-index:
    """

    generator: callable
    gen_kwargs: Dict[str, any] = OptionalField(default_factory=dict)
    caching: bool = False
    copying: bool = False

    def _get_initator(self):
        """Private method to get the correct initiator based on the streaming and caching attributes.

        Returns:
            function: The correct initiator function.
        """
        if self.caching:
            return Dataset.from_generator

        if self.copying:
            return CopyingReusableGenerator

        return ReusableGenerator

    def _get_stream(self):
        """Private method to get the stream based on the initiator function.

        Returns:
            object: The stream object.
        """
        return self._get_initator()(self.generator, gen_kwargs=self.gen_kwargs)

    def __iter__(self):
        return iter(self._get_stream())

    def peek(self):
        return next(iter(self))

    def take(self, n):
        for i, instance in enumerate(self):
            if i >= n:
                break
            yield instance


class MultiStream(dict):
    """A class for handling multiple streams of data in a dictionary-like format.

    This class extends dict and its values should be instances of the Stream class.

    Attributes:
        data (dict): A dictionary of Stream objects.
    """

    def __init__(self, data=None):
        """Initializes the MultiStream with the provided data.

        Args:
            data (dict, optional): A dictionary of Stream objects. Defaults to None.

        Raises:
            AssertionError: If the values are not instances of Stream or keys are not strings.
        """
        for key, value in data.items():
            isinstance(value, Stream), "MultiStream values must be Stream"
            isinstance(key, str), "MultiStream keys must be strings"
        super().__init__(data)

    def get_generator(self, key):
        """Gets a generator for a specified key.

        Args:
            key (str): The key for the generator.

        Yields:
            object: The next value in the stream.
        """
        yield from self[key]

    def set_caching(self, caching: bool):
        for stream in self.values():
            stream.caching = caching

    def set_copying(self, copying: bool):
        for stream in self.values():
            stream.copying = copying

    def to_dataset(self, disable_cache=True, cache_dir=None) -> DatasetDict:
        with tempfile.TemporaryDirectory() as dir_to_be_deleted:
            cache_dir = dir_to_be_deleted if disable_cache else cache_dir
            return DatasetDict(
                {
                    key: Dataset.from_generator(
                        self.get_generator,
                        keep_in_memory=disable_cache,
                        cache_dir=cache_dir,
                        gen_kwargs={"key": key},
                    )
                    for key in self.keys()
                }
            )

    def to_iterable_dataset(self) -> IterableDatasetDict:
        return IterableDatasetDict(
            {
                key: IterableDataset.from_generator(
                    self.get_generator, gen_kwargs={"key": key}
                )
                for key in self.keys()
            }
        )

    def __setitem__(self, key, value):
        assert isinstance(value, Stream), "StreamDict values must be Stream"
        assert isinstance(key, str), "StreamDict keys must be strings"
        super().__setitem__(key, value)

    @classmethod
    def from_generators(
        cls, generators: Dict[str, ReusableGenerator], caching=False, copying=False
    ):
        """Creates a MultiStream from a dictionary of ReusableGenerators.

        Args:
            generators (Dict[str, ReusableGenerator]): A dictionary of ReusableGenerators.
            caching (bool, optional): Whether the data should be cached or not. Defaults to False.
            copying (bool, optional): Whether the data should be copyied or not. Defaults to False.

        Returns:
            MultiStream: A MultiStream object.
        """
        assert all(isinstance(v, ReusableGenerator) for v in generators.values())
        return cls(
            {
                key: Stream(
                    generator.generator,
                    gen_kwargs=generator.gen_kwargs,
                    caching=caching,
                    copying=copying,
                )
                for key, generator in generators.items()
            }
        )

    @classmethod
    def from_iterables(
        cls, iterables: Dict[str, Iterable], caching=False, copying=False
    ):
        """Creates a MultiStream from a dictionary of iterables.

        Args:
            iterables (Dict[str, Iterable]): A dictionary of iterables.
            caching (bool, optional): Whether the data should be cached or not. Defaults to False.
            copying (bool, optional): Whether the data should be copyied or not. Defaults to False.

        Returns:
            MultiStream: A MultiStream object.
        """
        return cls(
            {
                key: Stream(
                    iterable.__iter__,
                    caching=caching,
                    copying=copying,
                )
                for key, iterable in iterables.items()
            }
        )
