from abc import abstractmethod
from dataclasses import field
from typing import Any, Dict, Generator, List, Optional, Union

from .artifact import Artifact
from .dataclass import NonPositionalField
from .random_utils import nested_seed
from .stream import MultiStream, Stream


class Operator(Artifact):
    pass


class OperatorError(Exception):
    def __init__(self, exception: Exception, operators: List[Operator]):
        super().__init__(
            "This error was raised by the following operators: "
            + ",\n".join([str(operator) for operator in operators])
            + "."
        )
        self.exception = exception
        self.operators = operators

    @classmethod
    def from_operator_error(cls, exception: Exception, operator: Operator):
        return cls(exception.exception, [*exception.operators, operator])

    @classmethod
    def from_exception(cls, exception: Exception, operator: Operator):
        return cls(exception, [operator])


class StreamingOperator(Artifact):
    """
    Base class for all stream operators in the streaming model.

    Stream operators are a key component of the streaming model and are responsible for processing continuous data streams.
    They perform operations such as transformations, aggregations, joins, windowing and more on these streams.
    There are several types of stream operators, including source operators, processing operators, etc.

    As a `StreamingOperator`, this class is responsible for performing operations on a stream, and must be implemented by all other specific types of stream operators in the system.
    When called, a `StreamingOperator` must return a MultiStream.

    As a subclass of `Artifact`, every `StreamingOperator` can be saved in a catalog for further usage or reference.

    """

    @abstractmethod
    def __call__(self, streams: Optional[MultiStream] = None) -> MultiStream:
        """
        Abstract method that performs operations on the stream.

        Args:
            streams (Optional[MultiStream]): The input MultiStream, which can be None.

        Returns:
            MultiStream: The output MultiStream resulting from the operations performed on the input.
        """


class StreamSource(StreamingOperator):
    """
    A class representing a stream source operator in the streaming system.

    A stream source operator is a special type of `StreamingOperator` that generates a data stream without taking any input streams. It serves as the starting point in a stream processing pipeline, providing the initial data that other operators in the pipeline can process.

    When called, a `StreamSource` should generate a `MultiStream`. This behavior must be implemented by any classes that inherit from `StreamSource`.

    """

    @abstractmethod
    def __call__(self) -> MultiStream:
        pass


class SourceOperator(StreamSource):
    """
    A class representing a source operator in the streaming system.

    A source operator is responsible for generating the data stream from some source, such as a database or a file. This is the starting point of a stream processing pipeline. The `SourceOperator` class is a type of `StreamSource`, which is a special type of `StreamingOperator` that generates an output stream but does not take any input streams.

    When called, a `SourceOperator` invokes its `process` method, which should be implemented by all subclasses to generate the required `MultiStream`.

    """

    caching: bool = NonPositionalField(default=None)

    def __call__(self) -> MultiStream:
        with nested_seed():
            multi_stream = self.process()
            if self.caching is not None:
                multi_stream.set_caching(self.caching)
            return multi_stream

    @abstractmethod
    def process(self) -> MultiStream:
        pass


class StreamInitializerOperator(StreamSource):
    """
    A class representing a stream initializer operator in the streaming system.

    A stream initializer operator is a special type of `StreamSource` that is capable of taking parameters during the stream generation process. This can be useful in situations where the stream generation process needs to be customized or configured based on certain parameters.

    When called, a `StreamInitializerOperator` invokes its `process` method, passing any supplied arguments and keyword arguments. The `process` method should be implemented by all subclasses to generate the required `MultiStream` based on the given arguments and keyword arguments.

    """

    caching: bool = NonPositionalField(default=None)

    def __call__(self, *args, **kwargs) -> MultiStream:
        with nested_seed():
            multi_stream = self.process(*args, **kwargs)
            if self.caching is not None:
                multi_stream.set_caching(self.caching)
            return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs) -> MultiStream:
        pass


class MultiStreamOperator(StreamingOperator):
    """
    A class representing a multi-stream operator in the streaming system.

    A multi-stream operator is a type of `StreamingOperator` that operates on an entire MultiStream object at once. It takes a `MultiStream` as input and produces a `MultiStream` as output. The `process` method should be implemented by subclasses to define the specific operations to be performed on the input `MultiStream`.
    """

    caching: bool = NonPositionalField(default=None)

    def __call__(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        with nested_seed():
            result = self._process_multi_stream(multi_stream)
            if self.caching is not None:
                result.set_caching(self.caching)
            return result

    def _process_multi_stream(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        result = self.process(multi_stream)
        assert isinstance(result, MultiStream), "MultiStreamOperator must return a MultiStream"
        return result

    @abstractmethod
    def process(self, multi_stream: MultiStream) -> MultiStream:
        pass


class SingleStreamOperator(MultiStreamOperator):
    """
    A class representing a single-stream operator in the streaming system.

    A single-stream operator is a type of `MultiStreamOperator` that operates on individual `Stream` objects within a `MultiStream`. It iterates through each `Stream` in the `MultiStream` and applies the `process` method. The `process` method should be implemented by subclasses to define the specific operations to be performed on each `Stream`.
    """

    def _process_multi_stream(self, multi_stream: MultiStream) -> MultiStream:
        result = {}
        for stream_name, stream in multi_stream.items():
            stream = self._process_single_stream(stream, stream_name)
            assert isinstance(stream, Stream), "SingleStreamOperator must return a Stream"
            result[stream_name] = stream

        return MultiStream(result)

    def _process_single_stream(self, stream: Stream, stream_name: str = None) -> Stream:
        return Stream(self._process_stream, gen_kwargs={"stream": stream, "stream_name": stream_name})

    def _process_stream(self, stream: Stream, stream_name: str = None) -> Generator:
        yield from self.process(stream, stream_name)

    @abstractmethod
    def process(self, stream: Stream, stream_name: str = None) -> Generator:
        pass


class PagedStreamOperator(SingleStreamOperator):
    """
    A class representing a paged-stream operator in the streaming system.

    A paged-stream operator is a type of `SingleStreamOperator` that operates on a page of instances in a `Stream` at a time, where a page is a subset of instances. The `process` method should be implemented by subclasses to define the specific operations to be performed on each page.
    """

    page_size: int = 1000

    def _process_stream(self, stream: Stream, stream_name: str = None) -> Generator:
        page = []
        for instance in stream:
            page.append(instance)
            if len(page) >= self.page_size:
                yield from self.process(page, stream_name)
                page = []
        yield from self.process(page, stream_name)

    @abstractmethod
    def process(self, page: List[Dict], stream_name: str = None) -> Generator:
        pass


class SingleStreamReducer(StreamingOperator):
    """
    A class representing a single-stream reducer in the streaming system.

    A single-stream reducer is a type of `StreamingOperator` that operates on individual `Stream` objects within a `MultiStream` and reduces each `Stream` to a single output value. The `process` method should be implemented by subclasses to define the specific reduction operation to be performed on each `Stream`.
    """

    def __call__(self, multi_stream: Optional[MultiStream] = None) -> Dict[str, Any]:
        result = {}
        for stream_name, stream in multi_stream.items():
            stream = self.process(stream)
            result[stream_name] = stream

        return result

    @abstractmethod
    def process(self, stream: Stream) -> Stream:
        pass


class StreamInstanceOperator(SingleStreamOperator):
    """
    A class representing a stream instance operator in the streaming system.

    A stream instance operator is a type of `SingleStreamOperator` that operates on individual instances within a `Stream`. It iterates through each instance in the `Stream` and applies the `process` method. The `process` method should be implemented by subclasses to define the specific operations to be performed on each instance.
    """

    def _process_stream(self, stream: Stream, stream_name: str = None) -> Generator:
        for instance in stream:
            yield self._process_instance(instance, stream_name)

    def _process_instance(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return self.process(instance, stream_name)

    @abstractmethod
    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        pass


class StreamInstanceOperatorValidator(StreamInstanceOperator):
    """
    A class representing a stream instance operator validator in the streaming system.

    A stream instance operator validator is a type of `StreamInstanceOperator` that includes a validation step. It operates on individual instances within a `Stream` and validates the result of processing each instance.
    """

    @abstractmethod
    def validate(self, instance):
        pass

    def _process_stream(self, stream: Stream, stream_name: str = None) -> Generator:
        iterator = iter(stream)
        first_instance = next(iterator)
        result = self._process_instance(first_instance, stream_name)
        self.validate(result)
        yield result
        yield from (self._process_instance(instance, stream_name) for instance in iterator)


class InstanceOperator(Artifact):
    """
    A class representing an instance operator in the streaming system.

    An instance operator is a type of `Artifact` that operates on a single instance (represented as a dict) at a time. It takes an instance as input and produces a transformed instance as output.
    """

    def __call__(self, data: dict) -> dict:
        return self.process(data)

    @abstractmethod
    def process(self, data: dict) -> dict:
        pass


class BaseFieldOperator(Artifact):
    """
    A class representing a field operator in the streaming system.

    A field operator is a type of `Artifact` that operates on a single field within an instance. It takes an instance and a field name as input, processes the field, and updates the field in the instance with the processed value.
    """

    def __call__(self, data: Dict[str, Any], field: str) -> dict:
        value = self.process(data[field])
        data[field] = value
        return data

    @abstractmethod
    def process(self, value: Any) -> Any:
        pass


class InstanceOperatorWithGlobalAccess(StreamingOperator):
    """
    A class representing an instance operator with global access in the streaming system.

    An instance operator with global access is a type of `StreamingOperator` that operates on individual instances within a `Stream` and can also access other streams.
    It uses the `accessible_streams` attribute to determine which other streams it has access to.
    In order to make this efficient and to avoid qudratic complexity, it caches the accessible streams by default.
    """

    accessible_streams: Union[MultiStream, List[str]] = None
    cache_accessible_streams: bool = True

    def __call__(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        result = {}

        if isinstance(self.accessible_streams, list):
            # cache the accessible streams:
            self.accessible_streams = MultiStream(
                {stream_name: multi_stream[stream_name] for stream_name in self.accessible_streams}
            )

        if self.cache_accessible_streams:
            for stream in self.accessible_streams.values():
                stream.caching = True

        for stream_name, stream in multi_stream.items():
            stream = Stream(self.generator, gen_kwargs={"stream": stream, "multi_stream": self.accessible_streams})
            result[stream_name] = stream

        return MultiStream(result)

    def generator(self, stream, multi_stream):
        yield from (self.process(instance, multi_stream) for instance in stream)

    @abstractmethod
    def process(self, instance: dict, multi_stream: MultiStream) -> dict:
        pass


class SequntialOperator(MultiStreamOperator):
    """
    A class representing a sequential operator in the streaming system.

    A sequential operator is a type of `MultiStreamOperator` that applies a sequence of other operators to a `MultiStream`. It maintains a list of `StreamingOperator`s and applies them in order to the `MultiStream`.
    """

    steps: List[StreamingOperator] = field(default_factory=list)

    def process(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        for operator in self.steps:
            multi_stream = operator(multi_stream)
        return multi_stream


class SourceSequntialOperator(SequntialOperator):
    """
    A class representing a source sequential operator in the streaming system.

    A source sequential operator is a type of `SequntialOperator` that starts with a source operator. The first operator in its list of steps is a `StreamSource`, which generates the initial `MultiStream` that the other operators then process.
    """

    def __call__(self) -> MultiStream:
        return super().__call__()

    def process(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        multi_stream = self.steps[0]()
        for operator in self.steps[1:]:
            multi_stream = operator(multi_stream)
        return multi_stream


class SequntialOperatorInitilizer(SequntialOperator):
    """
    A class representing a sequential operator initializer in the streaming system.

    A sequential operator initializer is a type of `SequntialOperator` that starts with a stream initializer operator. The first operator in its list of steps is a `StreamInitializerOperator`, which generates the initial `MultiStream` based on the provided arguments and keyword arguments.
    """

    def __call__(self, *args, **kwargs) -> MultiStream:
        with nested_seed():
            return self.process(*args, **kwargs)

    def process(self, *args, **kwargs) -> MultiStream:
        assert isinstance(
            self.steps[0], StreamInitializerOperator
        ), "The first step in a SequntialOperatorInitilizer must be a StreamInitializerOperator"
        multi_stream = self.steps[0](*args, **kwargs)
        for operator in self.steps[1:]:
            multi_stream = operator(multi_stream)
        return multi_stream
