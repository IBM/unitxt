import random
import typing
from dataclasses import field
from typing import Dict, List

from .artifact import Artifact
from .dataclass import AbstractField
from .random_utils import new_random_generator


class Collection(Artifact):
    items: typing.Collection = AbstractField()

    def __getitem__(self, key):
        try:
            return self.items[key]
        except LookupError as e:
            raise LookupError(f"Cannot find item {key!r} in {self!r}") from e


class ListCollection(Collection):
    items: List[Artifact] = field(default_factory=list)

    def __len__(self):
        return len(self.items)

    def append(self, item):
        self.items.append(item)

    def extend(self, other):
        self.items.extend(other.items)

    def __add__(self, other):
        return ListCollection(self.items.__add__(other.items))


class DictCollection(Collection):
    items: Dict[str, Artifact] = field(default_factory=dict)


class ItemPicker(Artifact):
    item: object = None

    def __call__(self, collection: Collection):
        try:
            return collection[int(self.item)]
        except (
            SyntaxError,
            KeyError,
            ValueError,
        ):  # in case picking from a dictionary
            return collection[self.item]


class RandomPicker(Artifact):
    random_generator: random.Random = field(
        default_factory=lambda: new_random_generator(sub_seed="random_picker")
    )

    def __call__(self, collection: Collection):
        if isinstance(collection, ListCollection):
            return self.random_generator.choice(list(collection.items))
        if isinstance(collection, DictCollection):
            return self.random_generator.choice(list(collection.items.values()))
        return None
