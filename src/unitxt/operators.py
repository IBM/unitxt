import importlib
import inspect
import uuid
from abc import abstractmethod
from copy import deepcopy
from dataclasses import field
from itertools import zip_longest
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from .artifact import Artifact, fetch_artifact
from .dataclass import NonPositionalField
from .dict_utils import dict_delete, dict_get, dict_set, is_subpath
from .operator import (
    MultiStream,
    MultiStreamOperator,
    PagedStreamOperator,
    SingleStreamOperator,
    SingleStreamReducer,
    StreamingOperator,
    StreamInitializerOperator,
    StreamInstanceOperator,
    StreamSource,
)
from .random_utils import random
from .stream import MultiStream, Stream
from .text_utils import nested_tuple_to_string
from .utils import flatten_dict


class FromIterables(StreamInitializerOperator):
    """
    Creates a MultiStream from iterables.

    Args:
        iterables (Dict[str, Iterable]): A dictionary where each key-value pair represents a stream name and its corresponding iterable.
    """

    def process(self, iterables: Dict[str, Iterable]) -> MultiStream:
        return MultiStream.from_iterables(iterables)


class IterableSource(StreamSource):
    iterables: Dict[str, Iterable]

    def __call__(self) -> MultiStream:
        return MultiStream.from_iterables(self.iterables)


class MapInstanceValues(StreamInstanceOperator):
    """A class used to map instance values into a stream.

    This class is a type of StreamInstanceOperator,
    it maps values of instances in a stream using predefined mappers.

    Attributes:
        mappers (Dict[str, Dict[str, str]]): The mappers to use for mapping instance values.
            Keys are the names of the fields to be mapped, and values are dictionaries
            that define the mapping from old values to new values.
        strict (bool): If True, the mapping is applied strictly. That means if a value
            does not exist in the mapper, it will raise a KeyError. If False, values
            that are not present in the mapper are kept as they are.
    """

    mappers: Dict[str, Dict[str, str]]
    strict: bool = True
    use_query = False

    def verify(self):
        # make sure the mappers are valid
        for key, mapper in self.mappers.items():
            assert isinstance(mapper, dict), f"Mapper for given field {key} should be a dict, got {type(mapper)}"
            for k, v in mapper.items():
                assert isinstance(k, str), f'Key "{k}" in mapper for field "{key}" should be a string, got {type(k)}'

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        for key, mapper in self.mappers.items():
            value = dict_get(instance, key, use_dpath=self.use_query)
            if value is not None:
                value = str(value)  # make sure the value is a string
                if self.strict:
                    dict_set(instance, key, mapper[value], use_dpath=self.use_query)
                else:
                    if value in mapper:
                        dict_set(instance, key, mapper[value], use_dpath=self.use_query)
        return instance


class FlattenInstances(StreamInstanceOperator):
    """
    Flattens each instance in a stream, making nested dictionary entries into top-level entries.

    Args:
        parent_key (str): A prefix to use for the flattened keys. Defaults to an empty string.
        sep (str): The separator to use when concatenating nested keys. Defaults to "_".
    """

    parent_key: str = ""
    sep: str = "_"

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return flatten_dict(instance, parent_key=self.parent_key, sep=self.sep)


class AddFields(StreamInstanceOperator):
    """
    Adds specified fields to each instance in a stream.

    Args:
        fields (Dict[str, object]): The fields to add to each instance.
    """

    fields: Dict[str, object]
    use_query: bool = False
    use_deepcopy: bool = False

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        if self.use_query:
            for key, value in self.fields.items():
                if self.use_deepcopy:
                    value = deepcopy(value)
                dict_set(instance, key, value, use_dpath=self.use_query)
        else:
            if self.use_deepcopy:
                self.fields = deepcopy(self.fields)
            instance.update(self.fields)
        return instance


class FieldOperator(StreamInstanceOperator):
    """
    A general stream that processes the values of a field (or multiple ones
    Args:
        field (Optional[str]): The field to process, if only a single one is passed Defaults to None
        to_field (Optional[str]): Field name to save, if only one field is to be saved, if None is passed the operator would happen in-place and replace "field" Defaults to None
        field_to_field (Optional[Union[List[Tuple[str, str]], Dict[str, str]]]): Mapping from fields to process to their names after this process, duplicates are allowed. Defaults to None
        process_every_value (bool): Processes the values in a list instead of the list as a value, similar to *var. Defaults to False
        use_query (bool): Whether to use dpath style queries. Defaults to False
    """

    field: Optional[str] = None
    to_field: Optional[str] = None
    field_to_field: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None
    process_every_value: bool = False
    use_query: bool = False
    get_default: Any = None
    not_exist_ok: bool = False

    def verify(self):
        super().verify()

        assert self.field is not None or self.field_to_field is not None, "Must supply a field to work on"
        assert (
            self.to_field is None or self.field_to_field is None
        ), f"Can not apply operator to create both on {self.to_field} and on the mapping from fields to fields {self.field_to_field}"
        assert (
            self.field is None or self.field_to_field is None
        ), f"Can not apply operator both on {self.field} and on the mapping from fields to fields {self.field_to_field}"
        assert self._field_to_field, f"the from and to fields must be defined got: {self._field_to_field}"

    @abstractmethod
    def process_value(self, value: Any) -> Any:
        pass

    def prepare(self):
        if self.to_field is None:
            self.to_field = self.field
        if self.field_to_field is None:
            self._field_to_field = [(self.field, self.to_field)]
        else:
            try:
                self._field_to_field = [(k, v) for k, v in self.field_to_field.items()]
            except AttributeError:
                self._field_to_field = self.field_to_field

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        for from_field, to_field in self._field_to_field:
            try:
                old_value = dict_get(
                    instance,
                    from_field,
                    use_dpath=self.use_query,
                    default=self.get_default,
                    not_exist_ok=self.not_exist_ok,
                )
            except TypeError as e:
                raise TypeError(f"Failed to get {from_field} from {instance}")
            if self.process_every_value:
                new_value = [self.process_value(value) for value in old_value]
            else:
                new_value = self.process_value(old_value)
            if self.use_query and is_subpath(from_field, to_field):
                dict_delete(instance, from_field)
            dict_set(instance, to_field, new_value, use_dpath=self.use_query, not_exist_ok=True)
        return instance


class RenameFields(FieldOperator):
    """
    Renames fields
    """

    def process_value(self, value: Any) -> Any:
        return value

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        res = super().process(instance=instance, stream_name=stream_name)
        vals = [x[1] for x in self._field_to_field]
        for key, _ in self._field_to_field:
            if self.use_query and "/" in key:
                continue
            if key not in vals:
                res.pop(key)
        return res


class AddConstant(FieldOperator):
    """
    Adds a number, similar to field + add
    Args:
        add (float): sum to add
    """

    add: float

    def process_value(self, value: Any) -> Any:
        return value + self.add


class ShuffleFieldValues(FieldOperator):
    """
    Shuffles an iterable value
    """

    def process_value(self, value: Any) -> Any:
        res = list(value)
        random.shuffle(res)
        return res


class JoinStr(FieldOperator):
    """
    Joins a list of strings (contents of a field), similar to str.join()
    Args:
        separator (str): text to put between values
    """

    separator: str = ","

    def process_value(self, value: Any) -> Any:
        return self.separator.join(str(x) for x in value)


class Apply(StreamInstanceOperator):
    __allow_unexpected_arguments__ = True
    function: Callable = NonPositionalField(required=True)
    to_field: str = NonPositionalField(required=True)

    def function_to_str(self, function: Callable) -> str:
        parts = []

        if hasattr(function, "__module__"):
            parts.append(function.__module__)
        if hasattr(function, "__qualname__"):
            parts.append(function.__qualname__)
        else:
            parts.append(function.__name__)

        result = ".".join(parts)

        return result

    def str_to_function(self, function_str: str) -> Callable:
        splitted = function_str.split(".", 1)
        if len(splitted) == 1:
            return __builtins__[module_name]
        else:
            module_name, function_name = splitted
            if module_name in __builtins__:
                obj = __builtins__[module_name]
            elif module_name in globals():
                obj = globals()[module_name]
            else:
                obj = importlib.import_module(module_name)
            for part in function_name.split("."):
                obj = getattr(obj, part)
            return obj

    def prepare(self):
        super().prepare()
        if isinstance(self.function, str):
            self.function = self.str_to_function(self.function)
        self._init_dict["function"] = self.function_to_str(self.function)

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        argv = [instance[arg] for arg in self._argv]
        kwargs = {key: instance[val] for key, val in self._kwargs}

        result = self.function(*argv, **kwargs)

        instance[self.to_field] = result
        return instance


class ListFieldValues(StreamInstanceOperator):
    """
    Concatanates values of multiple fields into a list to list(fields)
    """

    fields: str
    to_field: str
    use_query: bool = False

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        values = []
        for field in self.fields:
            values.append(dict_get(instance, field, use_dpath=self.use_query))
        instance[self.to_field] = values
        return instance


class ZipFieldValues(StreamInstanceOperator):
    """
    Zips values of multiple fields similar to list(zip(*fields))
    """

    fields: str
    to_field: str
    longest: bool = False
    use_query: bool = False

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        values = []
        for field in self.fields:
            values.append(dict_get(instance, field, use_dpath=self.use_query))
        if self.longest:
            zipped = zip_longest(*values)
        else:
            zipped = zip(*values)
        instance[self.to_field] = list(zipped)
        return instance


class IndexOf(StreamInstanceOperator):
    """
    Finds the location of one value in another (iterable) value similar to to_field=search_in.index(index_of)
    """

    search_in: str
    index_of: str
    to_field: str
    use_query: bool = False

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        lst = dict_get(instance, self.search_in, use_dpath=self.use_query)
        item = dict_get(instance, self.index_of, use_dpath=self.use_query)
        instance[self.to_field] = lst.index(item)
        return instance


class TakeByField(StreamInstanceOperator):
    """
    Takes value from one field based on another field similar to field[index]
    """

    field: str
    index: str
    to_field: str = None
    use_query: bool = False

    def prepare(self):
        if self.to_field is None:
            self.to_field = self.field

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        value = dict_get(instance, self.field, use_dpath=self.use_query)
        index_value = dict_get(instance, self.index, use_dpath=self.use_query)
        instance[self.to_field] = value[index_value]
        return instance


class CopyFields(FieldOperator):
    """
    Copies specified fields from one field to another.

    Args:
        field_to_field (Union[List[List], Dict[str, str]]): A list of lists, where each sublist contains the source field and the destination field, or a dictionary mapping source fields to destination fields.
        use_dpath (bool): Whether to use dpath for accessing fields. Defaults to False.
    """

    def process_value(self, value: Any) -> Any:
        return value


class AddID(StreamInstanceOperator):
    id_field_name: str = "id"

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        instance[self.id_field_name] = str(uuid.uuid4()).replace("-", "")
        return instance


class CastFields(StreamInstanceOperator):
    """
    Casts specified fields to specified types.

    Args:
        types (Dict[str, str]): A dictionary mapping fields to their new types.
        nested (bool): Whether to cast nested fields. Defaults to False.
        fields (Dict[str, str]): A dictionary mapping fields to their new types.
        defaults (Dict[str, object]): A dictionary mapping types to their default values for cases of casting failure.
    """

    types = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    fields: Dict[str, str] = field(default_factory=dict)
    failure_defaults: Dict[str, object] = field(default_factory=dict)
    use_nested_query: bool = False
    cast_multiple: bool = False

    def _cast_single(self, value, type, field):
        try:
            return self.types[type](value)
        except:
            if field not in self.failure_defaults:
                raise ValueError(
                    f'Failed to cast field "{field}" with value {value} to type "{type}", and no default value is provided.'
                )
            return self.failure_defaults[field]

    def _cast_multiple(self, values, type, field):
        values = [self._cast_single(value, type, field) for value in values]

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        for field, type in self.fields.items():
            value = dict_get(instance, field, use_dpath=self.use_nested_query)
            if self.cast_multiple:
                casted_value = self._cast_multiple(value, type, field)
            else:
                casted_value = self._cast_single(value, type, field)
            dict_set(instance, field, casted_value, use_dpath=self.use_nested_query)
        return instance


def recursive_divide(instance, divisor, strict=False):
    if isinstance(instance, dict):
        for key, value in instance.items():
            instance[key] = recursive_divide(value, divisor, strict=strict)
    elif isinstance(instance, list):
        for i, value in enumerate(instance):
            instance[i] = recursive_divide(value, divisor, strict=strict)
    elif isinstance(instance, float):
        instance /= divisor
    elif strict:
        raise ValueError(f"Cannot divide instance of type {type(instance)}")
    return instance


class DivideAllFieldsBy(StreamInstanceOperator):
    divisor: float = 1.0
    strict: bool = False
    recursive: bool = True

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return recursive_divide(instance, self.divisor, strict=self.strict)


class ArtifactFetcherMixin:
    """
    Provides a way to fetch and cache artifacts in the system.

    Args:
        cache (Dict[str, Artifact]): A cache for storing fetched artifacts.
    """

    cache: Dict[str, Artifact] = {}

    @classmethod
    def get_artifact(cls, artifact_identifier: str) -> Artifact:
        if artifact_identifier not in cls.cache:
            artifact, artifactory = fetch_artifact(artifact_identifier)
            cls.cache[artifact_identifier] = artifact
        return cls.cache[artifact_identifier]


class ApplyOperatorsField(StreamInstanceOperator, ArtifactFetcherMixin):
    """
    Applies value operators to each instance in a stream based on specified fields.

    Args:
        value_field (str): The field containing the value to be operated on.
        operators_field (str): The field containing the operators to be applied.
        default_operators (List[str]): A list of default operators to be used if no operators are found in the instance.
    """

    inputs_fields: str

    operators_field: str
    default_operators: List[str] = None
    fields_to_treat_as_list: List[str] = NonPositionalField(default_factory=list)

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        operator_names = instance.get(self.operators_field)
        if operator_names is None:
            assert (
                self.default_operators is not None
            ), f"No operators found in {self.field} field and no default operators provided"
            operator_names = self.default_operators

        if isinstance(operator_names, str):
            operator_names = [operator_names]

        for name in operator_names:
            operator = self.get_artifact(name)
            for field in self.inputs_fields:
                value = instance[field]
                if field in self.fields_to_treat_as_list:
                    instance[field] = [operator.process(v) for v in value]
                else:
                    instance[field] = operator.process(instance[field])

        return instance


class FilterByValues(SingleStreamOperator):
    """
    Filters a stream, yielding only instances that match specified values.

    Args:
        values (Dict[str, Any]): The values that instances should match to be included in the output.
    """

    values: Dict[str, Any]

    def process(self, stream: Stream, stream_name: str = None) -> Generator:
        for instance in stream:
            if all(instance[key] == value for key, value in self.values.items()):
                yield instance


class Unique(SingleStreamReducer):
    """
    Reduces a stream to unique instances based on specified fields.

    Args:
        fields (List[str]): The fields that should be unique in each instance.
    """

    fields: List[str] = field(default_factory=list)

    @staticmethod
    def to_tuple(instance: dict, fields: List[str]) -> tuple:
        result = []
        for field in fields:
            value = instance[field]
            if isinstance(value, list):
                value = tuple(value)
            result.append(value)
        return tuple(result)

    def process(self, stream: Stream) -> Stream:
        seen = set()
        for instance in stream:
            values = self.to_tuple(instance, self.fields)
            if values not in seen:
                seen.add(values)
        return list(seen)


class SplitByValue(MultiStreamOperator):
    """
    Splits a MultiStream into multiple streams based on unique values in specified fields.

    Args:
        fields (List[str]): The fields to use when splitting the MultiStream.
    """

    fields: List[str] = field(default_factory=list)

    def process(self, multi_stream: MultiStream) -> MultiStream:
        uniques = Unique(fields=self.fields)(multi_stream)

        result = {}

        for stream_name, stream in multi_stream.items():
            stream_unique_values = uniques[stream_name]
            for unique_values in stream_unique_values:
                filtering_values = {field: value for field, value in zip(self.fields, unique_values)}
                filtered_streams = FilterByValues(values=filtering_values)._process_single_stream(stream)
                filtered_stream_name = stream_name + "_" + nested_tuple_to_string(unique_values)
                result[filtered_stream_name] = filtered_streams

        return MultiStream(result)


class ApplyStreamOperatorsField(SingleStreamOperator, ArtifactFetcherMixin):
    """
    Applies stream operators to a stream based on specified fields in each instance.

    Args:
        field (str): The field containing the operators to be applied.
        reversed (bool): Whether to apply the operators in reverse order.
    """

    field: str
    reversed: bool = False

    def process(self, stream: Stream, stream_name: str = None) -> Generator:
        first_instance = stream.peak()

        operators = first_instance.get(self.field, [])
        if isinstance(operators, str):
            operators = [operators]

        if self.reversed:
            operators = list(reversed(operators))

        for operator_name in operators:
            operator = self.get_artifact(operator_name)
            assert isinstance(operator, StreamingOperator), f"Operator {operator_name} must be a SingleStreamOperator"

            stream = operator(MultiStream({"tmp": stream}))["tmp"]

        yield from stream


class AddFieldNamePrefix(StreamInstanceOperator):
    """
    Adds a prefix to each field name in each instance of a stream.

    Args:
        prefix_dict (Dict[str, str]): A dictionary mapping stream names to prefixes.
    """

    prefix_dict: Dict[str, str]

    def prepare(self):
        return super().prepare()

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return {self.prefix_dict[stream_name] + key: value for key, value in instance.items()}


class MergeStreams(MultiStreamOperator):
    """
    Merges multiple streams into a single stream.

    Args:
        new_stream_name (str): The name of the new stream resulting from the merge.
        add_origin_stream_name (bool): Whether to add the origin stream name to each instance.
        origin_stream_name_field_name (str): The field name for the origin stream name.
    """

    new_stream_name: str = "all"
    add_origin_stream_name: bool = True
    origin_stream_name_field_name: str = "origin"

    def merge(self, multi_stream):
        for stream_name, stream in multi_stream.items():
            for instance in stream:
                if self.add_origin_stream_name:
                    instance[self.origin_stream_name_field_name] = stream_name
                yield instance

    def process(self, multi_stream: MultiStream) -> MultiStream:
        return MultiStream({self.new_stream_name: Stream(self.merge, gen_kwargs={"multi_stream": multi_stream})})


class Shuffle(PagedStreamOperator):
    """
    Shuffles the order of instances in each page of a stream.

    Args:
        page_size (int): The size of each page in the stream. Defaults to 1000.
    """

    def process(self, page: List[Dict], stream_name: str = None) -> Generator:
        random.shuffle(page)
        yield from page


class EncodeLabels(StreamInstanceOperator):
    """
    Encode labels of specified fields together a into integers.

    Args:
        fields (List[str]): The fields to encode together.
    """

    fields: List[str]

    def _process_multi_stream(self, multi_stream: MultiStream) -> MultiStream:
        self.encoder = {}
        return super()._process_multi_stream(multi_stream)

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        for field in self.fields:
            values = dict_get(instance, field, use_dpath=True)
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if value not in self.encoder:
                    self.encoder[value] = len(self.encoder)
            new_values = [self.encoder[value] for value in values]
            dict_set(instance, field, new_values, use_dpath=True, set_multiple=True)

        return instance
