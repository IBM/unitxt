from abc import abstractmethod
from typing import Dict, Generator, List, Optional, Union

from .dataclass import NonPositionalField
from .operator import SourceOperator
from .random_utils import new_random_generator
from .stream import DynamicStream, MultiStream
from .type_utils import isoftype


class BaseFusion(SourceOperator):
    """BaseFusion operator that combines multiple multistreams into one.

    Args:
        origins: a dict of named SourceOperator objects (each to yield a MultiStream) or a list thereof,
          each is specified along with its input, so can generate a MultiStream
        include_splits: List of splits to include from each input MultiStream.
                If None, all splits are included.
    """

    origins: Union[List[SourceOperator], Dict[str, SourceOperator]]
    include_splits: Optional[List[str]] = NonPositionalField(default=None)

    @abstractmethod
    def fusion_generator(self, split) -> Generator:
        pass

    def prepare(self):
        assert isoftype(self.origins, Dict[str, SourceOperator]) or isoftype(
            self.origins, List[SourceOperator]
        )
        self.named_origins = (
            {i: self.origins[i]() for i in range(len(self.origins))}
            if isinstance(self.origins, list)
            else {name: origin() for name, origin in self.origins.items()}
        )

    def splits(self) -> List[str]:
        splits = []
        for _, origin in self.named_origins.items():
            for s in origin.keys():
                if s not in splits:
                    if self.include_splits is None or s in self.include_splits:
                        splits.append(s)
        return splits

    def process(
        self,
    ) -> MultiStream:
        result = {}
        for split in self.splits():
            result[split] = DynamicStream(
                self.fusion_generator, gen_kwargs={"split": split}
            )
        return MultiStream(result)


class FixedFusion(BaseFusion):
    """FixedFusion operator that combines multiple multistreams into one, limiting the number of instances taken from each split of each input multistream.

    Args:
        origins: Dict of named SourceOperator objects (each to yield a MultiStream), or a list thereof
        splits: List of splits (stream_names) to include, over all input multistreams. If None, all splits are included.
        max_instances_per_origin_split: Number of instances to take from each input split of each input multistream.
            If None, all instances of each split (that is specified in include_splits) are included in the result.

    """

    max_instances_per_origin_split: Optional[int] = None

    def prepare(self):
        super().prepare()

    # flake8: noqa: C901
    def fusion_generator(self, split) -> Generator:
        for origin_name, origin in self.named_origins.items():
            if split not in origin:
                continue
            emitted_from_this_split = 0
            for instance in origin[split]:
                if (
                    self.max_instances_per_origin_split is not None
                    and emitted_from_this_split >= self.max_instances_per_origin_split
                ):
                    break
                if isinstance(origin_name, str):
                    # named origins, not anonymous, record in instance
                    if "group" in instance:
                        instance["group"] = origin_name + "/" + instance["group"]
                    else:
                        instance["group"] = origin_name
                emitted_from_this_split += 1
                yield instance


class WeightedFusion(BaseFusion):
    """Fusion operator that combines multiple MultiStream-s.

    Args:
        origins: Dict of named MultiStream objects, or a list thereof
        weights: Dict of named weights for each origin, or a list thereof
        max_total_examples: Total number of instances to return per returned split.
            If None, all instances are returned
    """

    origins: Union[Dict[str, SourceOperator], List[SourceOperator]] = None
    weights: Union[Dict[str, Union[float, int]], List[Union[int, float]]] = None
    max_total_examples: int = None
    ignore_origin_groups: List[str] = ["unitxt"]

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
        assert isoftype(self.weights, Dict[str, Union[int, float]]) or isoftype(
            self.weights, List[Union[int, float]]
        )
        assert isinstance(self.origins, dict) == isinstance(self.weights, dict)

    def prepare(self):
        super().prepare()
        self.named_weights = (
            {i: float(self.weights[i]) for i in range(len(self.weights))}
            if isinstance(self.weights, list)
            else {k: float(v) for (k, v) in self.weights.items()}
        )

    def fusion_generator(self, split) -> Generator:
        iterators = {
            named_origin: iter(origin[split])
            for named_origin, origin in self.named_origins.items()
        }
        total_examples = 0
        random_generator = new_random_generator(sub_seed="weighted_fusion_" + split)
        while (
            self.max_total_examples is None or total_examples < self.max_total_examples
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
                    if (
                        "group" in instance
                        and instance["group"] not in self.ignore_origin_groups
                    ):
                        instance["group"] = origin_name + "/" + instance["group"]
                    else:
                        instance["group"] = origin_name
                total_examples += 1
                yield instance

            except StopIteration:
                iterators.pop(origin_name)
