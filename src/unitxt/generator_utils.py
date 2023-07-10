import inspect
import itertools


class ReusableGenerator:
    def __init__(self, generator, gen_argv=[], gen_kwargs={}):
        self._generator = generator
        self._gen_kwargs = gen_kwargs
        self._gen_argv = gen_argv

    def get_generator(self):
        return self._generator

    def get_gen_kwargs(self):
        return self._gen_kwargs

    def construct(self):
        return self._generator(*self._gen_argv, **self._gen_kwargs)

    def __iter__(self):
        return iter(self.construct())

    def __call__(self):
        yield from self.construct()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._generator.__name__}, gen_argv={self._gen_argv}, gen_kwargs={self._gen_kwargs})"


if __name__ == "__main__":
    from itertools import chain, islice

    # Creating objects of MyIterable
    iterable1 = ReusableGenerator(range, gen_argv=[1, 4])
    iterable2 = ReusableGenerator(range, gen_argv=[4, 7])

    # Using itertools.chain
    chained = list(chain(iterable1, iterable2))
    print(chained)  # Prints: [1, 2, 3, 4, 5, 6]

    # Using itertools.islice
    sliced = list(islice(ReusableGenerator(range, gen_argv=[1, 7]), 1, 4))
    print(sliced)  # Prints: [2, 3, 4]

    # now same test with generators
    def generator(start, end):
        for i in range(start, end):
            yield i

    iterable1 = ReusableGenerator(generator, gen_argv=[1, 4])
    iterable2 = ReusableGenerator(generator, gen_argv=[4, 7])

    chained = list(chain(iterable1, iterable2))
    print(chained)  # Prints: [1, 2, 3, 4, 5, 6]

    sliced = list(islice(ReusableGenerator(generator, gen_argv=[1, 7]), 1, 4))
    print(sliced)  # Prints: [2, 3, 4]
