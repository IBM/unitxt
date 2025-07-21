from itertools import zip_longest
from typing import Any, Dict, Generator, List, Optional

from .dict_utils import dict_get, dict_set
from .operator import InstanceOperator
from .operators import FieldOperator, StreamOperator
from .stream import Stream
from .utils import recursive_shallow_copy


class Dictify(FieldOperator):
    with_keys: List[str]

    def process_value(self, tup: Any) -> Any:
        return dict(zip(self.with_keys, tup))


class Zip(InstanceOperator):
    fields: List[str]
    to_field: str

    def zip(self, values):
        return list(zip(*values))

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        values = []
        for field in self.fields:
            values.append(dict_get(instance, field))
        dict_set(instance, self.to_field, self.zip(values))
        return instance


class ZipLongest(Zip):
    fields: List[str]
    fill_value: Any = None

    def zip(self, values):
        return list(zip_longest(*values, fillvalue=self.fill_value))


class DictToTuplesList(FieldOperator):
    def process_value(self, dic: Dict) -> Any:
        return list(dic.items())


def flatten(container):
    def _flat_gen(x):
        for item in x:
            if isinstance(item, (list, tuple)):
                yield from _flat_gen(item)
            else:
                yield item

    return type(container)(_flat_gen(container))


class Flatten(FieldOperator):
    def process_value(self, value: Any) -> Any:
        return flatten(value)


class Wrap(FieldOperator):
    inside: str

    def verify(self):
        super().verify()
        if self.inside not in ["list", "tuple", "set"]:
            raise ValueError(
                f"Wrap.inside support only types: [list, tuple, set], got {self.inside}"
            )

    def process_value(self, value: Any) -> Any:
        if self.inside == "list":
            return [value]
        if self.inside == "tuple":
            return (value,)
        return {
            value,
        }


class Chunk(FieldOperator):
    size: int

    def process_value(self, collection: Any) -> Any:
        return [
            collection[i : i + self.size] for i in range(0, len(collection), self.size)
        ]


class Slice(FieldOperator):
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    def process_value(self, collection: Any) -> Any:
        slicer = slice(self.start, self.stop, self.step)
        return collection[slicer]


class Get(FieldOperator):
    item: Any

    def process_value(self, collection: Any) -> Any:
        return collection[self.item]


class Pop(FieldOperator):
    item: Any = None

    def process_value(self, collection: Any) -> Any:
        return collection.pop(self.item)


class DuplicateByList(StreamOperator):
    field: str
    to_field: Optional[str] = None
    use_deep_copy: bool = False

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        to_field = self.field if self.to_field is None else self.to_field
        for instance in stream:
            elements = dict_get(instance, self.field)
            for element in elements:
                if self.use_deep_copy:
                    instance_copy = recursive_shallow_copy(instance)

                else:
                    instance_copy = instance.copy()
                dict_set(instance_copy, to_field, element)
                yield instance_copy


class Explode(DuplicateByList):
    pass


class DuplicateBySubLists(StreamOperator):
    field: str
    to_field: Optional[str] = None
    use_deep_copy: bool = False
    start: int = 1
    end: int = 0
    step: int = 1

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        to_field = self.field if self.to_field is None else self.to_field
        for instance in stream:
            elements = dict_get(instance, self.field)
            end = len(elements) + 1 + self.end
            for i in range(self.start, end, self.step):
                if self.use_deep_copy:
                    instance_copy = recursive_shallow_copy(instance)
                    instance_copy[to_field] = elements[:i]
                else:
                    instance_copy = {
                        **instance,
                        self.field: elements,
                        to_field: elements[:i],
                    }
                yield instance_copy


class ExplodeSubLists(DuplicateBySubLists):
    pass


class GetLength(FieldOperator):
    def process_value(self, collection: Any) -> Any:
        return len(collection)


class Filter(FieldOperator):
    values: List[Any]

    def process_value(self, collection: Any) -> Any:
        # If collection is a list, tuple, or set
        if isinstance(collection, (list, set, tuple)):
            return type(collection)(
                item for item in collection if item not in self.values
            )

        # If collection is a dictionary, filter by keys
        if isinstance(collection, dict):
            return {k: v for k, v in collection.items() if k not in self.values}

        # If collection is of an unsupported type
        raise TypeError(f"Unsupported collection type: {type(collection)}")
