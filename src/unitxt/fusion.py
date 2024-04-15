from abc import abstractmethod
from typing import Dict, Generator, List, Optional, Union

from .dataclass import NonPositionalField
from .operator import SourceOperator
from .random_utils import new_random_generator
from .stream import MultiStream, Stream
from .type_utils import isoftype


class BaseFusion(SourceOperator):
    """BaseFusion operator that combines multiple streams into one.

    Args:
        origins: a dict of named SourceOperators, or a list of such sources
        include_splits: List of splits to include from each SourceOperator.
                If None, all splits are included.
    """

    origins: Union[List[SourceOperator], Dict[str, SourceOperator]]
    include_splits: Optional[List[str]] = NonPositionalField(default=None)

    @abstractmethod
    def fusion_generator(self, split) -> Generator:
        pass

    def prepare(self):
        super().prepare()
        assert isoftype(self.origins, Dict[str, SourceOperator]) or isoftype(
            self.origins, List[SourceOperator]
        )
        self.named_origins = (
            {i: self.origins[i] for i in range(len(self.origins))}
            if isinstance(self.origins, list)
            else self.origins
        )

    def splits(self) -> List[str]:
        splits = []
        for _, origin in self.named_origins.items():
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
        origins: Dict of named SourceOperator objects, or a list thereof
        examples_per_task: Number of examples per task. If None, all examples are returned.
        splits: List of splits to include. If None, all splits are included.
    """

    max_instances_per_origin: Optional[int] = None

    def prepare(self):
        super().prepare()

    # flake8: noqa: C901
    def fusion_generator(self, split) -> Generator:
        for origin_name, origin in self.named_origins.items():
            multi_stream = origin()
            if split not in multi_stream:
                continue
            iterator = iter(multi_stream[split])
            if self.max_instances_per_origin is not None:
                for _ in range(self.max_instances_per_origin):
                    try:
                        instance = next(iterator)
                        if isinstance(origin_name, int):
                            yield instance
                        if "group" in instance:
                            instance["group"] = origin_name + "/" + instance["group"]
                        else:
                            instance["group"] = origin_name
                        yield instance
                    except StopIteration:
                        break
            else:
                for instance in iterator:
                    if "group" in instance:
                        instance["group"] = origin_name + "/" + instance["group"]
                    else:
                        instance["group"] = origin_name
                    yield instance


class WeightedFusion(BaseFusion):
    """Fusion operator that combines multiple streams based.

    Args:
        origins: Dict of named of SourceOperator objects, or a list thereof
        weights: Dict of named of weights for each origin, or a list thereof
        max_total_examples: Total number of examples to return. If None, all examples are returned.
    """

    origins: Union[Dict[str, SourceOperator], List[SourceOperator]] = None
    weights: Union[Dict[str, float], List[float]] = None
    max_total_examples: int = None

    def verify(self):
        super().verify()
        assert self.origins is not None, "origins must be specified"
        assert self.weights is not None, "weights must be specified"
        assert len(self.origins) == len(
            self.weights
        ), "origins and weights must have the same length"
        assert isoftype(self.origins, Dict[str, SourceOperator]) or isoftype(
            self.origins, List[SourceOperator]
        )
        assert isoftype(self.weights, Dict[str, float]) or isoftype(
            self.weights, List[float]
        )
        assert isinstance(self.origins, dict) == isinstance(self.weights, dict)

    def prepare(self):
        super().prepare()
        self.named_weights = (
            {i: self.weights[i] for i in range(len(self.weights))}
            if isinstance(self.weights, list)
            else self.weights
        )

    def fusion_generator(self, split) -> Generator:
        iterators = {
            named_origin: iter(origin()[split])
            for named_origin, origin in self.named_origins.items()
        }
        total_examples = 0
        random_generator = new_random_generator(sub_seed="weighted_fusion_" + split)
        while (
            self.max_total_examples is None or total_examples <= self.max_total_examples
        ) and len(iterators) > 0:
            population = list(iterators.keys())
            origin_name = random_generator.choices(
                population=population,
                weights=[self.named_weights[name] for name in population],
            )[0]
            iterator = iterators[origin_name]
            try:
                instance = next(iterator)
                if isinstance(origin_name, str):
                    if "group" in instance:
                        instance["group"] = origin_name + "/" + instance["group"]
                    else:
                        instance["group"] = origin_name
                total_examples += 1
                yield instance

            except StopIteration:
                iterators.pop(origin_name)
