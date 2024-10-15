from typing import Any, Generator, List, Optional

from .dict_utils import dict_get, dict_set
from .operators import FieldOperator, StreamOperator
from .stream import Stream
from .utils import recursive_shallow_copy


class Dictify(FieldOperator):
    with_keys: List[str]

    def process_value(self, tup: Any) -> Any:
        return dict(zip(self.with_keys, tup))


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

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        to_field = self.field if self.to_field is None else self.to_field
        for instance in stream:
            elements = instance[self.field]
            for i in range(1, len(elements) + 1):
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
