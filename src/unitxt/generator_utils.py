from typing import Any, Dict, List

from .dataclass import Dataclass, OptionalField
from .utils import recursive_copy


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
    def __iter__(self):
        for instance in self.activate():
            yield recursive_copy(instance)
