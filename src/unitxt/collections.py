import typing
from dataclasses import field
from typing import Dict, List

from .artifact import Artifact
from .dataclass import AbstractField
from .random_utils import random


class Collection(Artifact):
    items: typing.Collection = AbstractField()

    def __getitem__(self, key):
        try:
            return self.items[key]
        except LookupError:
            raise LookupError(f"Cannot find item {repr(key)} in {repr(self)}")


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
        except (SyntaxError, KeyError, ValueError) as e:  # in case picking from a dictionary
            return collection[self.item]


class RandomPicker(Artifact):
    def __call__(self, collection: Collection):
        if isinstance(collection, ListCollection):
            return random.choice(list(collection.items))
        elif isinstance(collection, DictCollection):
            return random.choice(list(collection.items.values()))
