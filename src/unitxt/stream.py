import tempfile
import traceback
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Generator, Iterable, List

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from .dataclass import Dataclass, OptionalField
from .generator_utils import CopyingReusableGenerator, ReusableGenerator
from .logging_utils import get_logger
from .settings_utils import get_settings

settings = get_settings()
logger = get_logger()


class Stream(Dataclass):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def peek(self):
        pass

    @abstractmethod
    def take(self, n):
        pass

    @abstractmethod
    def set_copying(self, copying: bool):
        pass


class ListStream(Stream):
    instances_list: List[Dict[str, Any]]
    copying: bool = False

    def __iter__(self):
        if self.copying:
            return iter(deepcopy(self.instances_list))
        return iter(self.instances_list)

    def peek(self):
        return next(iter(self))

    def take(self, n) -> Generator:
        for i, instance in enumerate(self.instances_list):
            if i >= n:
                break
            yield instance

    def set_copying(self, copying: bool):
        self.copying = copying


class GeneratorStream(Stream):
    """A class for handling streaming data in a customizable way.

    This class provides methods for generating, caching, and manipulating streaming data.

    Attributes:
        generator (function): A generator function for streaming data. :no-index:
        gen_kwargs (dict, optional): A dictionary of keyword arguments for the generator function. :no-index:
        caching (bool): Whether the data is cached or not. :no-index:
    """

    generator: Callable
    gen_kwargs: Dict[str, Any] = OptionalField(default_factory=dict)
    caching: bool = False
    copying: bool = False

    def _get_initiator(self):
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
        return self._get_initiator()(self.generator, gen_kwargs=self.gen_kwargs)

    def __iter__(self):
        return iter(self._get_stream())

    def peek(self):
        return next(iter(self))

    def take(self, n):
        for i, instance in enumerate(self):
            if i >= n:
                break
            yield instance

    def set_copying(self, copying: bool):
        self.copying = copying


class FaultyStreamError(Exception):
    """Base class for all stream-related exceptions."""

    pass


class MissingStreamError(FaultyStreamError):
    """Raised when a required stream is missing."""

    pass


class EmptyStreamError(FaultyStreamError):
    """Raised when a stream is unexpectedly empty."""

    pass


def eager_failed():
    traceback.print_exc()
    warnings.warn(
        "The eager execution has failed due to the error above.", stacklevel=2
    )


class DynamicStream(Stream):
    generator: Callable
    gen_kwargs: Dict[str, Any] = OptionalField(default_factory=dict)
    caching: bool = False
    copying: bool = False

    def __post_init__(self):
        self.stream = None
        if settings.use_eager_execution:
            try:
                instances_list = []
                for instance in self.generator(**self.gen_kwargs):
                    instances_list.append(instance)
                self.stream = ListStream(
                    instances_list=instances_list, copying=self.copying
                )
            except FaultyStreamError:
                eager_failed()
            except RuntimeError as e:
                if isinstance(e.__cause__, FaultyStreamError):
                    eager_failed()
                else:
                    raise e

        if self.stream is None:
            self.stream = GeneratorStream(
                generator=self.generator,
                gen_kwargs=self.gen_kwargs,
                caching=self.caching,
                copying=self.copying,
            )

    def __iter__(self):
        return self.stream.__iter__()

    def peek(self):
        return self.stream.peek()

    def take(self, n):
        return self.stream.take(n)

    def set_copying(self, copying: bool):
        self.stream.set_copying(copying)


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

    def get_generator(self, key) -> Generator:
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
            stream.set_copying(copying)

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
            copying (bool, optional): Whether the data should be copied or not. Defaults to False.

        Returns:
            MultiStream: A MultiStream object.
        """
        assert all(isinstance(v, ReusableGenerator) for v in generators.values())
        return cls(
            {
                key: DynamicStream(
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
            copying (bool, optional): Whether the data should be copied or not. Defaults to False.

        Returns:
            MultiStream: A MultiStream object.
        """
        return cls(
            {
                key: DynamicStream(
                    iterable.__iter__,
                    caching=caching,
                    copying=copying,
                )
                for key, iterable in iterables.items()
            }
        )
