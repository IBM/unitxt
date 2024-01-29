import copy
from abc import abstractmethod
from typing import Generator, List, Optional

from .dataclass import NonPositionalField
from .operator import SourceOperator
from .random_utils import new_random_generator
from .stream import MultiStream, Stream


class BaseFusion(SourceOperator):
    """BaseFusion operator that combines multiple streams into one.

    Args:
        include_splits: List of splits to include. If None, all splits are included.
    """

    origins: List[SourceOperator]
    include_splits: Optional[List[str]] = NonPositionalField(default=None)

    @abstractmethod
    def fusion_generator(self, split) -> Generator:
        pass

    def splits(self) -> Generator:
        splits = []
        for origin in self.origins:
            for s in origin().keys():
                if s not in splits:
                    if self.include_splits is None or s in self.include_splits:
                        splits.append(s)
        return splits

    def process(
        self,
    ) -> MultiStream:
        result = {}
        for split in self.splits():
            result[split] = Stream(self.fusion_generator, gen_kwargs={"split": split})
        return MultiStream(result)


class FixedFusion(BaseFusion):
    """FixedFusion operator that combines multiple streams into one based on a fixed number of examples per task.

    Args:
        orgins: List of SourceOperator objects.
        examples_per_task: Number of examples per task. If None, all examples are returned.
        splits: List of splits to include. If None, all splits are included.
    """

    max_instances_per_origin: Optional[int] = None

    def fusion_generator(self, split) -> Generator:
        for origin in self.origins:
            iterator = iter(origin()[split])
            if self.max_instances_per_origin is not None:
                for _ in range(self.max_instances_per_origin):
                    try:
                        yield next(iterator)
                    except StopIteration:
                        break
            else:
                yield from iterator


class WeightedFusion(BaseFusion):
    """Fusion operator that combines multiple streams based.

    Args:
        orgins: List of SourceOperator objects.
        weights: List of weights for each origin.
        max_total_examples: Total number of examples to return. If None, all examples are returned.
    """

    origins: List[SourceOperator] = None
    weights: List[float] = None
    max_total_examples: int = None

    def verify(self):
        super().verify()
        assert self.origins is not None, "origins must be specified"
        assert self.weights is not None, "weights must be specified"
        assert len(self.origins) == len(
            self.weights
        ), "origins and weights must have the same length"

    def fusion_generator(self, split) -> Generator:
        weights = copy.deepcopy(self.weights)
        iterators = [iter(origin()[split]) for origin in self.origins]
        total_examples = 0
        random_generator = new_random_generator(sub_seed="weighted_fusion_" + split)
        while (
            self.max_total_examples is None or total_examples <= self.max_total_examples
        ) and len(iterators) > 0:
            iterator = random_generator.choices(population=iterators, weights=weights)[
                0
            ]
            try:
                yield next(iterator)
                total_examples += 1
            except StopIteration:
                index = iterators.index(iterator)
                iterators.pop(index)
                weights.pop(index)
