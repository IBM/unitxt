from .stream import MultiStream, Stream
from .artifact import Artifact

from abc import abstractmethod
from typing import Optional, List, Dict, Generator, Union, Any
from dataclasses import field


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
    @abstractmethod
    def __call__(self, streams: Optional[MultiStream] = None) -> MultiStream:
        pass


class StreamSource(StreamingOperator):
    @abstractmethod
    def __call__(self) -> MultiStream:
        pass


class SourceOperator(StreamSource):
    def __call__(self) -> MultiStream:
        return self.process()

    @abstractmethod
    def process(self) -> MultiStream:
        pass


class StreamInitializerOperator(StreamSource):
    def __call__(self, *args, **kwargs) -> MultiStream:
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs) -> MultiStream:
        pass


class MultiStreamOperator(StreamingOperator):
    def __call__(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        return self._process_multi_stream(multi_stream)

    def _process_multi_stream(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        result = self.process(multi_stream)
        assert isinstance(result, MultiStream), "MultiStreamOperator must return a MultiStream"
        return result

    @abstractmethod
    def process(self, multi_stream: MultiStream) -> MultiStream:
        pass


class SingleStreamOperator(MultiStreamOperator):
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


# class StreamGeneratorOperator(SingleStreamOperator):

#     def stream(self, stream):
#         return Stream(self.process, gen_kwargs={'stream': stream})

#     @abstractmethod
#     def process(self, stream: Stream) -> Generator:
#         yield None


class SingleStreamReducer(StreamingOperator):
    def __call__(self, multi_stream: Optional[MultiStream] = None) -> Dict[str, Any]:
        result = {}
        for stream_name, stream in multi_stream.items():
            stream = self.process(stream)
            result[stream_name] = stream

        return result

    @abstractmethod
    def process(self, stream: Stream) -> Any:
        pass


class StreamInstanceOperator(SingleStreamOperator):
    def _process_stream(self, stream: Stream, stream_name: str = None) -> Generator:
        for instance in stream:
            yield self._process_instance(instance, stream_name)

    def _process_instance(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return self.process(instance, stream_name)

    @abstractmethod
    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        pass


class StreamInstanceOperatorValidator(StreamInstanceOperator):
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
    def __call__(self, data: dict) -> dict:
        return self.process(data)

    @abstractmethod
    def process(self, data: dict) -> dict:
        pass


class FieldOperator(Artifact):
    def __call__(self, data: Dict[str, Any], field: str) -> dict:
        value = self.process(data[field])
        data[field] = value
        return data

    @abstractmethod
    def process(self, value: Any) -> Any:
        pass


# class NamedStreamInstanceOperator(StreamingOperator):

#     def __call__(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
#         result = {}
#         for stream_name, stream in multi_stream.items():
#             stream = Stream(self.generator, gen_kwargs={'stream': stream, 'stream_name': stream_name})
#             result[stream_name] = stream
#         return MultiStream(result)

#     def verify_first_instance(self, instance):
#         pass

#     def generator(self, stream, stream_name):
#         iterator = iter(stream)
#         first_instance = next(iterator)
#         result = self.process(first_instance, stream_name)
#         self.verify_first_instance(result)
#         yield result
#         yield from (self.process(instance) for instance in iterator)

#     @abstractmethod
#     def process(self, instance: dict, stream_name: str) -> dict:
#         pass


class InstanceOperatorWithGlobalAccess(StreamingOperator):
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
                stream.set_caching(True)

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
    steps: List[StreamingOperator] = field(default_factory=list)

    def process(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        for operator in self.steps:
            multi_stream = operator(multi_stream)
        return multi_stream


class SourceSequntialOperator(SequntialOperator):
    def __call__(self) -> MultiStream:
        return super().__call__()

    def process(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        multi_stream = self.steps[0]()
        for operator in self.steps[1:]:
            multi_stream = operator(multi_stream)
        return multi_stream


class SequntialOperatorInitilizer(SequntialOperator):
    def __call__(self, *args, **kwargs) -> MultiStream:
        return self.process(*args, **kwargs)

    def process(self, *args, **kwargs) -> MultiStream:
        assert isinstance(
            self.steps[0], StreamInitializerOperator
        ), "The first step in a SequntialOperatorInitilizer must be a StreamInitializerOperator"
        multi_stream = self.steps[0](*args, **kwargs)
        for operator in self.steps[1:]:
            multi_stream = operator(multi_stream)
        return multi_stream
