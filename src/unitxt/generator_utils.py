import copy
import inspect
import itertools
from typing import Any, Dict, List, Optional

from .dataclass import Dataclass, OptionalField


class ReusableGenerator(Dataclass):
    generator: callable
    gen_argv: List[Any] = OptionalField(default_factory=list)
    gen_kwargs: Dict[str, Any] = OptionalField(default_factory=dict)

    def activate(self):
        return self.generator(*self.gen_argv, **self.gen_kwargs)

    def __iter__(self):
        yield from self.activate()

    def __call__(self):
        yield from iter(self)


class CopyingReusableGenerator(ReusableGenerator):
    copying: bool = True

    def __iter__(self):
        for instance in self.activate():
            yield copy.deepcopy(instance)


class MemoryCachingReusableGenerator(ReusableGenerator):

    caching: bool = True
    copying = False
    cache: Optional[List[Any]] = None

    def __iter__(self):
        if self.cache is None:
            self.cache = []
            for instance in super().__iter__():
                self.cache.append(instance)
                yield instance
        else:
            yield from self.cache


class CopyingMemoryCachingReusableGenerator(MemoryCachingReusableGenerator):
    copying: bool = True
