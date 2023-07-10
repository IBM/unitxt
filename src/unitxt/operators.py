from .stream import MultiStream, Stream
from .artifact import Artifact, fetch_artifact
from .operator import (
    StreamInstanceOperator,
    MultiStreamOperator,
    SingleStreamOperator,
    SingleStreamReducer,
    StreamInitializerOperator,
    Stream,
    MultiStream,
)

from dataclasses import field
from typing import List, Union, Dict, Optional, Generator, Any, Iterable

from typing import Dict, Any


class FromIterables(StreamInitializerOperator):
    def process(self, iterables: Dict[str, Iterable]) -> MultiStream:
        return MultiStream.from_iterables(iterables)


class MapInstanceValues(StreamInstanceOperator):
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


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class FlattenInstances(StreamInstanceOperator):
    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return flatten_dict(instance)


class AddFields(StreamInstanceOperator):
    fields: Dict[str, object]

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return {**instance, **self.fields}


class ArtifactFetcherMixin:
    cache: Dict[str, Artifact] = {}

    @classmethod
    def get_artifact(cls, artifact_identifier: str) -> Artifact:
        if artifact_identifier not in cls.cache:
            artifact, artifactory = fetch_artifact(artifact_identifier)
            cls.cache[artifact_identifier] = artifact
        return cls.cache[artifact_identifier]


class ApplyValueOperatorsField(StreamInstanceOperator, ArtifactFetcherMixin):
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
    values: Dict[str, Any]

    def process(self, stream: Stream, stream_name: str = None) -> Generator:
        for instance in stream:
            if all(instance[key] == value for key, value in self.values.items()):
                yield instance


class Unique(SingleStreamReducer):
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


from .text_utils import nested_tuple_to_string


class SplitByValue(MultiStreamOperator):
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
    prefix_dict: Dict[str, str]

    def prepare(self):
        return super().prepare()

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return {self.prefix_dict[stream_name] + key: value for key, value in instance.items()}


class MergeStreams(MultiStreamOperator):
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
