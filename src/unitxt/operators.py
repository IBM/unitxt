from dataclasses import field
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

from .text_utils import nested_tuple_to_string
from .artifact import Artifact, fetch_artifact
from .operator import (
    MultiStream,
    MultiStreamOperator,
    SingleStreamOperator,
    SingleStreamReducer,
    Stream,
    StreamInitializerOperator,
    StreamInstanceOperator,
    PagedStreamOperator,
)
from .stream import MultiStream, Stream
from .utils import flatten_dict
import random
from .utils import dict_query


class FromIterables(StreamInitializerOperator):
    """
    Creates a MultiStream from iterables.

    Args:
        iterables (Dict[str, Iterable]): A dictionary where each key-value pair represents a stream name and its corresponding iterable.
    """
    def process(self, iterables: Dict[str, Iterable]) -> MultiStream:
        return MultiStream.from_iterables(iterables)


class MapInstanceValues(StreamInstanceOperator):
    """
    Maps values in each instance of a stream based on the provided mappers.

    Args:
        mappers (Dict[str, Dict[str, str]]): A dictionary where each key-value pair represents a field in the instance and a mapper for that field.
        strict (bool): If True, the operator will raise a KeyError if a value is not in its corresponding mapper. If False, unmapped values will be left unchanged. Defaults to True.
    """
    mappers: Dict[str, Dict[str, str]]
    strict: bool = True

    def verify(self):
        # make sure the mappers are valid
        for key, mapper in self.mappers.items():
            assert isinstance(mapper, dict), f"Mapper for given field {key} should be a dict, got {type(mapper)}"
            for k, v in mapper.items():
                assert isinstance(k, str), f'Key "{k}" in mapper for field "{key}" should be a string, got {type(k)}'

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        result = {}
        for key, value in instance.items():
            str_value = str(value)
            if key in self.mappers:
                mapper = self.mappers[key]
                if self.strict:
                    value = mapper[str_value]
                else:
                    if str_value in mapper:
                        value = mapper[str_value]
            result[key] = value
        return result


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

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        instance.update(self.fields)
        return instance


class MapNestedDictValuesByQueries(StreamInstanceOperator):
    field_to_query: Dict[str, str]

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        updates = {}
        for field, query in self.field_to_query.items():
            updates[field] = dict_query(instance, query)
        instance.update(updates)
        return instance


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


class ApplyValueOperatorsField(StreamInstanceOperator, ArtifactFetcherMixin):
    """
    Applies value operators to each instance in a stream based on specified fields.

    Args:
        value_field (str): The field containing the value to be operated on.
        operators_field (str): The field containing the operators to be applied.
        default_operators (List[str]): A list of default operators to be used if no operators are found in the instance.
    """
    value_field: str
    operators_field: str
    default_operators: List[str] = None

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
            instance = operator(instance, self.value_field)

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
            assert isinstance(
                operator, SingleStreamOperator
            ), f"Operator {operator_name} must be a SingleStreamOperator"
            stream = operator.process(stream)

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