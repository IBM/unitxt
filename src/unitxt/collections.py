from .artifact import Artifact

from abc import abstractmethod

from dataclasses import field
import random


class Collection(Artifact):
    @abstractmethod
    def __getitem__(self, key):
        pass


class ListCollection(Collection):
    items: list = field(default_factory=list)

    def __getitem__(self, index):
        return self.items[index]


class DictCollection(Collection):
    items: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.items[key]


class ItemPicker(Artifact):
    item: object = None

    def __call__(self, collection: Collection):
        return collection[self.item]


class RandomPicker(Artifact):
    def __call__(self, collection: Collection):
        if isinstance(collection, ListCollection):
            return random.choice(list(collection.items))
        elif isinstance(collection, DictCollection):
            return random.choice(list(collection.items.values()))
