from abc import abstractmethod
from dataclasses import field
from typing import Any, Dict, Generator, List, Optional, Union

from pkg_resources import DistributionNotFound, VersionConflict, require

from .artifact import Artifact
from .dataclass import FinalField, InternalField, NonPositionalField
from .error_utils import error_context
from .settings_utils import get_constants
from .stream import DynamicStream, EmptyStreamError, MultiStream, Stream

constants = get_constants()


class Operator(Artifact):
    pass


class PackageRequirementsMixin(Artifact):
    """Base class used to automatically check for the existence of required Python dependencies for an artifact (e.g., Operator or Metric).

    The _requirements_list is either a list of required packages or a dictionary mapping required packages to installation instructions.
    The _requirements_list should be used at class level definition, and the requirements at instance creation.

    - **List format**: Just specify the package names, optionally with version annotations (e.g., ["torch>=1.2.4", "numpy<1.19"]).
    - **Dict format**: Specify package names as keys and installation instructions as values
      (e.g., {"torch>=1.2.4": "Install torch with `pip install torch>=1.2.4`"}).

    When a package version annotation is specified (like `torch>=1.2.4`), the `check_missing_requirements` method
    verifies that the installed version meets the specified constraint.
    """

    _requirements_list: Union[List[str], Dict[str, str]] = InternalField(
        default_factory=list
    )
    requirements: Union[List[str], Dict[str, str]] = FinalField(
        also_positional=False, default_factory=list
    )

    def prepare(self):
        self.check_missing_requirements(self._requirements_list)
        self.check_missing_requirements(self.requirements)
        super().prepare()

    def check_missing_requirements(self, requirements=None):
        if requirements is None:
            requirements = self._requirements_list
        if isinstance(requirements, list):
            requirements = {package: "" for package in requirements}

        missing_packages = []
        version_mismatched_packages = []
        installation_instructions = []

        for package, installation_instruction in requirements.items():
            try:
                # Use pkg_resources.require to verify the package requirement
                require(package)
            except DistributionNotFound:
                missing_packages.append(package)
                installation_instructions.append(
                    installation_instruction
                    or f"Install {package} with `pip install {package}`"
                )
            except VersionConflict as e:
                version_mismatched_packages.append(
                    f"{package} (installed: {e.dist.version}, required: {e.req})"
                )
                installation_instructions.append(
                    installation_instruction
                    or f"Update {package} to the required version with `pip install '{package}'`"
                )

        if missing_packages or version_mismatched_packages:
            raise MissingRequirementsError(
                self.__class__.__name__,
                missing_packages,
                version_mismatched_packages,
                installation_instructions,
            )


class MissingRequirementsError(Exception):
    def __init__(
        self,
        class_name,
        missing_packages,
        version_mismatched_packages,
        installation_instructions,
    ):
        self.class_name = class_name
        self.missing_packages = missing_packages
        self.version_mismatched_packages = version_mismatched_packages
        self.installation_instructions = installation_instructions

        missing_message = (
            f"Missing package(s): {', '.join(self.missing_packages)}."
            if self.missing_packages
            else ""
        )
        version_message = (
            f"Version mismatch(es): {', '.join(self.version_mismatched_packages)}."
            if self.version_mismatched_packages
            else ""
        )

        self.message = (
            f"{self.class_name} requires the following dependencies:\n"
            f"{missing_message}\n{version_message}\n"
            + "\n".join(self.installation_instructions)
        )
        super().__init__(self.message)


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


class StreamingOperator(Operator, PackageRequirementsMixin):
    """Base class for all stream operators in the streaming model.

    Stream operators are a key component of the streaming model and are responsible for processing continuous data streams.
    They perform operations such as transformations, aggregations, joins, windowing and more on these streams.
    There are several types of stream operators, including source operators, processing operators, etc.

    As a `StreamingOperator`, this class is responsible for performing operations on a stream, and must be implemented by all other specific types of stream operators in the system.
    When called, a `StreamingOperator` must return a MultiStream.

    As a subclass of `Artifact`, every `StreamingOperator` can be saved in a catalog for further usage or reference.

    """

    @abstractmethod
    def __call__(self, streams: Optional[MultiStream] = None) -> MultiStream:
        """Abstract method that performs operations on the stream.

        Args:
            streams (Optional[MultiStream]): The input MultiStream, which can be None.

        Returns:
            MultiStream: The output MultiStream resulting from the operations performed on the input.
        """


class SideEffectOperator(StreamingOperator):
    """Base class for operators that does not affect the stream."""

    def __call__(self, streams: Optional[MultiStream] = None) -> MultiStream:
        self.process()
        return streams

    @abstractmethod
    def process() -> None:
        pass


def instance_generator(instance):
    yield instance


def stream_single(instance: Dict[str, Any]) -> Stream:
    return DynamicStream(
        generator=instance_generator, gen_kwargs={"instance": instance}
    )


class MultiStreamOperator(StreamingOperator):
    """A class representing a multi-stream operator in the streaming system.

    A multi-stream operator is a type of `StreamingOperator` that operates on an entire MultiStream object at once. It takes a `MultiStream` as input and produces a `MultiStream` as output. The `process` method should be implemented by subclasses to define the specific operations to be performed on the input `MultiStream`.
    """

    caching: bool = NonPositionalField(default=None)

    def __call__(
        self, multi_stream: Optional[MultiStream] = None, **instance: Dict[str, Any]
    ) -> Union[MultiStream, Dict[str, Any]]:
        self.before_process_multi_stream()
        if instance:
            if multi_stream is not None:
                return self.process_instance(instance)
        result = self._process_multi_stream(multi_stream)
        if self.caching is not None:
            result.set_caching(self.caching)
        return result

    def before_process_multi_stream(self):
        pass

    def _process_multi_stream(
        self, multi_stream: Optional[MultiStream] = None
    ) -> MultiStream:
        result = self.process(multi_stream)
        assert isinstance(
            result, MultiStream
        ), "MultiStreamOperator must return a MultiStream"
        return result

    @abstractmethod
    def process(self, multi_stream: MultiStream) -> MultiStream:
        pass

    def process_instance(self, instance, stream_name=constants.instance_stream):
        instance = self.verify_instance(instance)
        multi_stream = MultiStream({stream_name: stream_single(instance)})
        processed_multi_stream = self(multi_stream)
        return instance_result(processed_multi_stream[stream_name])


class SourceOperator(MultiStreamOperator):
    """A class representing a source operator in the streaming system.

    A source operator is responsible for generating the data stream from some source, such as a database or a file.
    This is the starting point of a stream processing pipeline.
    The ``SourceOperator`` class is a type of ``MultiStreamOperator``, which is a special type of ``StreamingOperator``
    that generates an output stream but does not take any input streams.

    When called, a ``SourceOperator`` invokes its ``process`` method, which should be implemented by all subclasses
    to generate the required ``MultiStream``.

    """

    def _process_multi_stream(
        self, multi_stream: Optional[MultiStream] = None
    ) -> MultiStream:
        result = self.process()
        assert isinstance(
            result, MultiStream
        ), "MultiStreamOperator must return a MultiStream"
        return result

    @abstractmethod
    def process(self) -> MultiStream:
        pass

    def get_splits(self):
        return list(self.process().keys())


class StreamInitializerOperator(SourceOperator):
    """A class representing a stream initializer operator in the streaming system.

    A stream initializer operator is a special type of ``SourceOperator`` that is capable
    of taking parameters during the stream generation process.
    This can be useful in situations where the stream generation process needs to be
    customized or configured based on certain parameters.

    When called, a ``StreamInitializerOperator`` invokes its ``process`` method, passing any supplied
    arguments and keyword arguments. The ``process`` method should be implemented by all subclasses
    to generate the required ``MultiStream`` based on the given arguments and keyword arguments.

    """

    caching: bool = NonPositionalField(default=None)

    def __call__(self, *args, **kwargs) -> MultiStream:
        multi_stream = self.process(*args, **kwargs)
        if self.caching is not None:
            multi_stream.set_caching(self.caching)
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs) -> MultiStream:
        pass


def instance_result(result_stream):
    result = list(result_stream)
    if len(result) == 0:
        return None
    if len(result) == 1:
        return result[0]
    return result


class StreamOperator(MultiStreamOperator):
    """A class representing a single-stream operator in the streaming system.

    A single-stream operator is a type of ``MultiStreamOperator`` that operates on individual
    ``Stream`` objects within a ``MultiStream``. It iterates through each ``Stream`` in the ``MultiStream``
    and applies the ``process`` method.

    The ``process`` method should be implemented by subclasses to define the specific operations
    to be performed on each ``Stream``.

    """

    apply_to_streams: List[str] = NonPositionalField(
        default=None
    )  # None apply to all streams
    dont_apply_to_streams: List[str] = NonPositionalField(default=None)

    def _process_multi_stream(self, multi_stream: MultiStream) -> MultiStream:
        result = {}
        for stream_name, stream in multi_stream.items():
            if self._is_should_be_processed(stream_name):
                stream = self._process_single_stream(stream, stream_name)
            else:
                stream = stream
            assert isinstance(stream, Stream), "StreamOperator must return a Stream"
            result[stream_name] = stream

        return MultiStream(result)

    def _process_single_stream(
        self, stream: Stream, stream_name: Optional[str] = None
    ) -> Stream:
        return DynamicStream(
            self._process_stream,
            gen_kwargs={"stream": stream, "stream_name": stream_name},
        )

    def _is_should_be_processed(self, stream_name):
        if (
            self.apply_to_streams is not None
            and self.dont_apply_to_streams is not None
            and stream_name in self.apply_to_streams
            and stream_name in self.dont_apply_to_streams
        ):
            raise ValueError(
                f"Stream '{stream_name}' can be in either apply_to_streams or dont_apply_to_streams not both."
            )

        return (
            self.apply_to_streams is None or stream_name in self.apply_to_streams
        ) and (
            self.dont_apply_to_streams is None
            or stream_name not in self.dont_apply_to_streams
        )

    def _process_stream(
        self, stream: Stream, stream_name: Optional[str] = None
    ) -> Generator:
        with error_context(self, stream=stream_name):
            yield from self.process(stream, stream_name)

    @abstractmethod
    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        pass

    def process_instance(self, instance, stream_name=constants.instance_stream):
        instance = self.verify_instance(instance)
        processed_stream = self._process_single_stream(
            stream_single(instance), stream_name
        )
        return instance_result(processed_stream)


class SingleStreamOperator(StreamOperator):
    pass


class PagedStreamOperator(StreamOperator):
    """A class representing a paged-stream operator in the streaming system.

    A paged-stream operator is a type of ``StreamOperator`` that operates on a page of instances
    in a ``Stream`` at a time, where a page is a subset of instances.
    The ``process`` method should be implemented by subclasses to define the specific operations
    to be performed on each page.

    Args:
        page_size (int):
            The size of each page in the stream. Defaults to 1000.

    """

    page_size: int = 1000

    def _process_stream(
        self, stream: Stream, stream_name: Optional[str] = None
    ) -> Generator:
        page = []
        page_number = 0
        for instance in stream:
            page.append(instance)
            if len(page) >= self.page_size:
                with error_context(
                    self,
                    stream=stream_name,
                    page=page_number,
                    page_size=len(page),
                ):
                    yield from self.process(page, stream_name)
                page = []
                page_number += 1
        if page:  # Handle any remaining instances in the last partial page
            with error_context(
                self,
                stream=stream_name,
                page=page_number,
                page_size=len(page),
                final_page=True,
            ):
                yield from self._process_page(page, stream_name)

    def _process_page(
        self, page: List[Dict], stream_name: Optional[str] = None
    ) -> Generator:
        yield from self.process(page, stream_name)

    @abstractmethod
    def process(self, page: List[Dict], stream_name: Optional[str] = None) -> Generator:
        pass

    def process_instance(self, instance, stream_name=constants.instance_stream):
        instance = self.verify_instance(instance)
        processed_stream = self._process_page([instance], stream_name)
        return instance_result(processed_stream)


class SingleStreamReducer(StreamingOperator):
    """A class representing a single-stream reducer in the streaming system.

    A single-stream reducer is a type of ``StreamingOperator`` that operates on individual
    ``Stream`` objects within a ``MultiStream`` and reduces each ``Stream`` to a single output value.

    The ``process`` method should be implemented by subclasses to define the specific reduction operation
    to be performed on each ``Stream``.

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


class InstanceOperator(StreamOperator):
    """A class representing a stream instance operator in the streaming system.

    A stream instance operator is a type of ``StreamOperator`` that operates on individual instances
    within a ``Stream``. It iterates through each instance in the ``Stream`` and applies the ``process`` method.
    The ``process`` method should be implemented by subclasses to define the specific operations
    to be performed on each instance.
    """

    def _process_stream(
        self, stream: Stream, stream_name: Optional[str] = None
    ) -> Generator:
        for _index, instance in enumerate(stream):
            with error_context(self, stream=stream_name, instance=_index):
                yield self._process_instance(instance, stream_name)

    def _process_instance(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance = self.verify_instance(instance)
        return self.process(instance, stream_name)

    @abstractmethod
    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        pass

    def process_instance(self, instance, stream_name=constants.instance_stream):
        return self._process_instance(instance, stream_name)


class InstanceOperatorValidator(InstanceOperator):
    """A class representing a stream instance operator validator in the streaming system.

    A stream instance operator validator is a type of ``InstanceOperator`` that includes a validation step.
    It operates on individual instances within a ``Stream`` and validates the result of processing each instance.
    """

    @abstractmethod
    def validate(self, instance):
        pass

    def _process_stream(
        self, stream: Stream, stream_name: Optional[str] = None
    ) -> Generator:
        iterator = iter(stream)
        try:
            first_instance = next(iterator)
        except StopIteration as e:
            raise EmptyStreamError(f"Stream '{stream_name}' is empty") from e
        result = self._process_instance(first_instance, stream_name)
        self.validate(result, stream_name)
        yield result
        yield from (
            self._process_instance(instance, stream_name) for instance in iterator
        )


class InstanceOperatorWithMultiStreamAccess(StreamingOperator):
    """A class representing an instance operator with global access in the streaming system.

    An instance operator with global access is a type of `StreamingOperator` that operates on individual instances within a `Stream` and can also access other streams.
    It uses the `accessible_streams` attribute to determine which other streams it has access to.
    In order to make this efficient and to avoid qudratic complexity, it caches the accessible streams by default.
    """

    def __call__(
        self, multi_stream: Optional[MultiStream] = None, **instance: Dict[str, Any]
    ) -> MultiStream:
        if instance:
            raise NotImplementedError("Instance mode is not supported")

        result = {}

        for stream_name, stream in multi_stream.items():
            stream = DynamicStream(
                self.generator,
                gen_kwargs={"stream": stream, "multi_stream": multi_stream},
            )
            result[stream_name] = stream

        return MultiStream(result)

    def generator(self, stream, multi_stream):
        yield from (
            self.process(self.verify_instance(instance), multi_stream)
            for instance in stream
        )

    @abstractmethod
    def process(self, instance: dict, multi_stream: MultiStream) -> dict:
        pass


class SequentialMixin(Artifact):
    max_steps: Optional[int] = None
    steps: List[StreamingOperator] = field(default_factory=list)

    def num_steps(self) -> int:
        return len(self.steps)

    def set_max_steps(self, max_steps):
        assert (
            max_steps <= self.num_steps()
        ), f"Max steps requested ({max_steps}) is larger than defined steps {self.num_steps()}"
        assert max_steps >= 1, f"Max steps requested ({max_steps}) is less than 1"
        self.max_steps = max_steps

    def get_last_step_description(self):
        last_step = (
            self.max_steps - 1 if self.max_steps is not None else len(self.steps) - 1
        )
        return self.steps[last_step].__description__

    def _get_max_steps(self):
        return self.max_steps if self.max_steps is not None else len(self.steps)


class SequentialOperator(MultiStreamOperator, SequentialMixin):
    """A class representing a sequential operator in the streaming system.

    A sequential operator is a type of `MultiStreamOperator` that applies a sequence of other operators to a
    `MultiStream`. It maintains a list of `StreamingOperator`s and applies them in order to the `MultiStream`.
    """

    def process(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        for operator in self.steps[0 : self._get_max_steps()]:
            multi_stream = operator(multi_stream)
        return multi_stream


class SourceSequentialOperator(SourceOperator, SequentialMixin):
    """A class representing a source sequential operator in the streaming system.

    A source sequential operator is a type of `SequentialOperator` that starts with a source operator.
    The first operator in its list of steps is a `SourceOperator`, which generates the initial `MultiStream`
    that the other operators then process.
    """

    def process(self, multi_stream: Optional[MultiStream] = None) -> MultiStream:
        assert (
            self.num_steps() > 0
        ), "Calling process on a SourceSequentialOperator without any steps"
        multi_stream = self.steps[0]()
        for operator in self.steps[1 : self._get_max_steps()]:
            multi_stream = operator(multi_stream)
        return multi_stream


class SequentialOperatorInitializer(SequentialOperator):
    """A class representing a sequential operator initializer in the streaming system.

    A sequential operator initializer is a type of `SequntialOperator` that starts with a stream initializer operator. The first operator in its list of steps is a `StreamInitializerOperator`, which generates the initial `MultiStream` based on the provided arguments and keyword arguments.
    """

    def __call__(self, *args, **kwargs) -> MultiStream:
        return self.process(*args, **kwargs)

    def process(self, *args, **kwargs) -> MultiStream:
        assert (
            self.num_steps() > 0
        ), "Calling process on a SequentialOperatorInitializer without any steps"

        assert isinstance(
            self.steps[0], StreamInitializerOperator
        ), "The first step in a SequentialOperatorInitializer must be a StreamInitializerOperator"
        multi_stream = self.steps[0](*args, **kwargs)
        for operator in self.steps[1 : self._get_max_steps()]:
            multi_stream = operator(multi_stream)
        return multi_stream
