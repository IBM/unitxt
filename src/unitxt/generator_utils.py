import copy
from typing import Any, Dict, List

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
    def __iter__(self):
        for instance in self.activate():
            yield copy.deepcopy(instance)


# if __name__ == "__main__":
#     from itertools import chain, islice

#     # Creating objects of MyIterable
#     iterable1 = ReusableGenerator(range, gen_argv=[1, 4])
#     iterable2 = ReusableGenerator(range, gen_argv=[4, 7])

#     # Using itertools.chain
#     chained = list(chain(iterable1, iterable2))
#     logger.info(chained)  # Prints: [1, 2, 3, 4, 5, 6]

#     # Using itertools.islice
#     sliced = list(islice(ReusableGenerator(range, gen_argv=[1, 7]), 1, 4))
#     logger.info(sliced)  # Prints: [2, 3, 4]

#     # now same test with generators
#     def generator(start, end):
#         for i in range(start, end):
#             yield i

#     iterable1 = ReusableGenerator(generator, gen_argv=[1, 4])
#     iterable2 = ReusableGenerator(generator, gen_argv=[4, 7])

#     chained = list(chain(iterable1, iterable2))
#     logger.info(chained)  # Prints: [1, 2, 3, 4, 5, 6]

#     sliced = list(islice(ReusableGenerator(generator, gen_argv=[1, 7]), 1, 4))
#     logger.info(sliced)  # Prints: [2, 3, 4]
