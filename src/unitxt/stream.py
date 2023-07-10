from .generator_utils import ReusableGenerator

from typing import Iterable, Dict

from datasets import IterableDatasetDict, IterableDataset, DatasetDict, Dataset


class Stream:
    """A class for handling streaming data in a customizable way.

    This class provides methods for generating, caching, and manipulating streaming data.

    Attributes:
        generator (function): A generator function for streaming data.
        gen_kwargs (dict, optional): A dictionary of keyword arguments for the generator function.
        streaming (bool): Whether the data is streaming or not.
        caching (bool): Whether the data is cached or not.
    """

    def __init__(self, generator, gen_kwargs=None, streaming=True, caching=False):
        """Initializes the Stream with the provided parameters.

        Args:
            generator (function): A generator function for streaming data.
            gen_kwargs (dict, optional): A dictionary of keyword arguments for the generator function. Defaults to None.
            streaming (bool, optional): Whether the data is streaming or not. Defaults to True.
            caching (bool, optional): Whether the data is cached or not. Defaults to False.
        """

        self.generator = generator
        self.gen_kwargs = gen_kwargs if gen_kwargs is not None else {}
        self.streaming = streaming
        self.caching = caching

    def _get_initator(self):
        """Private method to get the correct initiator based on the streaming and caching attributes.

        Returns:
            function: The correct initiator function.
        """
        if self.streaming:
            if self.caching:
                return IterableDataset.from_generator
            else:
                return ReusableGenerator
        else:
            if self.caching:
                return Dataset.from_generator
            else:
                raise ValueError("Cannot create non-streaming non-caching stream")

    def _get_stream(self):
        """Private method to get the stream based on the initiator function.

        Returns:
            object: The stream object.
        """
        return self._get_initator()(self.generator, gen_kwargs=self.gen_kwargs)

    def set_caching(self, caching):
        self.caching = caching

    def set_streaming(self, streaming):
        self.streaming = streaming

    def __iter__(self):
        return iter(self._get_stream())

    def unwrap(self):
        return self._get_stream()

    def peak(self):
        return next(iter(self))

    def take(self, n):
        for i, instance in enumerate(self):
            if i >= n:
                break
            yield instance

    def __repr__(self):
        return f"{self.__class__.__name__}(generator={self.generator.__name__}, gen_kwargs={self.gen_kwargs}, streaming={self.streaming}, caching={self.caching})"


def is_stream(obj):
    return isinstance(obj, IterableDataset) or isinstance(obj, Stream) or isinstance(obj, Dataset)


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

    def unwrap(self, cls):
        return cls({key: value.unwrap() for key, value in self.items()})

    def to_dataset(self) -> DatasetDict:
        return DatasetDict(
            {key: Dataset.from_generator(self.get_generator, gen_kwargs={"key": key}) for key in self.keys()}
        )

    def to_iterable_dataset(self) -> IterableDatasetDict:
        return IterableDatasetDict(
            {key: IterableDataset.from_generator(self.get_generator, gen_kwargs={"key": key}) for key in self.keys()}
        )

    def __setitem__(self, key, value):
        assert isinstance(value, Stream), "StreamDict values must be Stream"
        assert isinstance(key, str), "StreamDict keys must be strings"
        super().__setitem__(key, value)

    @classmethod
    def from_generators(cls, generators: Dict[str, ReusableGenerator], streaming=True, caching=False):
        """Creates a MultiStream from a dictionary of ReusableGenerators.

        Args:
            generators (Dict[str, ReusableGenerator]): A dictionary of ReusableGenerators.
            streaming (bool, optional): Whether the data should be streaming or not. Defaults to True.
            caching (bool, optional): Whether the data should be cached or not. Defaults to False.

        Returns:
            MultiStream: A MultiStream object.
        """

        assert all(isinstance(v, ReusableGenerator) for v in generators.values())
        return cls(
            {
                key: Stream(
                    generator.get_generator(),
                    gen_kwargs=generator.get_gen_kwargs(),
                    streaming=streaming,
                    caching=caching,
                )
                for key, generator in generators.items()
            }
        )

    @classmethod
    def from_iterables(cls, iterables: Dict[str, Iterable], streaming=True, caching=False):
        """Creates a MultiStream from a dictionary of iterables.

        Args:
            iterables (Dict[str, Iterable]): A dictionary of iterables.
            streaming (bool, optional): Whether the data should be streaming or not. Defaults to True.
            caching (bool, optional): Whether the data should be cached or not. Defaults to False.

        Returns:
            MultiStream: A MultiStream object.
        """

        return cls(
            {
                key: Stream(iterable.__iter__, gen_kwargs={}, streaming=streaming, caching=caching)
                for key, iterable in iterables.items()
            }
        )
