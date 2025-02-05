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
        subsets: a dict of named SourceOperator objects (each to yield a MultiStream) or a list thereof,
          each is specified along with its input, so can generate a MultiStream
        include_splits: List of splits to include from each input MultiStream.
                If None, all splits are included.
    """

    subsets: Union[List[SourceOperator], Dict[str, SourceOperator]]
    include_splits: Optional[List[str]] = NonPositionalField(default=None)

    @abstractmethod
    def fusion_generator(self, split) -> Generator:
        pass

    def prepare_subsets(self):
        assert isoftype(self.subsets, Dict[str, SourceOperator]) or isoftype(
            self.subsets, List[SourceOperator]
        )
        self.named_subsets = {}
        if isinstance(self.subsets, list):
            for i in range(len(self.subsets)):
                self.named_subsets[i] = self.subsets[i]
        else:
            for name, origin in self.subsets.items():
                try:
                    self.named_subsets[name] = origin
                except Exception as e:
                    raise RuntimeError(f"Exception in subset: {name}") from e

    def splits(self) -> List[str]:
        self.prepare_subsets()
        if self.include_splits is not None:
            return self.include_splits
        return ["train", "test", "validation"]

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
        subsets: Dict of named SourceOperator objects (each to yield a MultiStream), or a list thereof
        splits: List of splits (stream_names) to include, over all input multistreams. If None, all splits are included.
        max_instances_per_subset: Number of instances to take from each input split of each input multistream.
            If None, all instances of each split (that is specified in include_splits) are included in the result.

    """

    max_instances_per_subset: Optional[int] = None

    def prepare(self):
        super().prepare()

    # flake8: noqa: C901
    def fusion_generator(self, split) -> Generator:
        for origin_name, origin in self.named_subsets.items():
            multi_stream = origin()
            if split not in multi_stream:
                continue
            emitted_from_this_split = 0
            try:
                for instance in multi_stream[split]:
                    if (
                        self.max_instances_per_subset is not None
                        and emitted_from_this_split >= self.max_instances_per_subset
                    ):
                        break
                    if isinstance(origin_name, str):
                        if "subset" not in instance:
                            instance["subset"] = []
                        instance["subset"].insert(0, origin_name)
                    emitted_from_this_split += 1
                    yield instance
            except Exception as e:
                raise RuntimeError(f"Exception in subset: {origin_name}") from e


class WeightedFusion(BaseFusion):
    """Fusion operator that combines multiple MultiStream-s.

    Args:
        subsets: Dict of named MultiStream objects, or a list thereof
        weights: Dict of named weights for each origin, or a list thereof
        max_total_examples: Total number of instances to return per returned split.
            If None, all instances are returned
    """

    subsets: Union[Dict[str, SourceOperator], List[SourceOperator]] = None
    weights: Union[Dict[str, Union[float, int]], List[Union[int, float]]] = None
    max_total_samples: int = None

    def verify(self):
        super().verify()
        assert self.subsets is not None, "subsets must be specified"
        assert self.weights is not None, "weights must be specified"
        assert len(self.subsets) == len(
            self.weights
        ), "subsets and weights must have the same length"
        assert isoftype(self.subsets, Dict[str, SourceOperator]) or isoftype(
            self.subsets, List[SourceOperator]
        )
        assert isoftype(self.weights, Dict[str, Union[int, float]]) or isoftype(
            self.weights, List[Union[int, float]]
        )
        assert isinstance(self.subsets, dict) == isinstance(self.weights, dict)

    def prepare(self):
        super().prepare()
        self.named_weights = (
            {i: float(self.weights[i]) for i in range(len(self.weights))}
            if isinstance(self.weights, list)
            else {k: float(v) for (k, v) in self.weights.items()}
        )

    def fusion_generator(self, split) -> Generator:
        iterators = {}
        for origin_name, origin in self.named_subsets.items():
            multi_stream = origin()
            if split not in multi_stream:
                continue
            iterators[origin_name] = iter(multi_stream[split])
        total_examples = 0
        random_generator = new_random_generator(sub_seed="weighted_fusion_" + split)
        while (
            self.max_total_samples is None or total_examples < self.max_total_samples
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
                    if "subset" not in instance:
                        instance["subset"] = []
                    instance["subset"].insert(0, origin_name)
                total_examples += 1
                yield instance

            except StopIteration:
                iterators.pop(origin_name)
            except Exception as e:
                raise RuntimeError(f"Exception in subset: {origin_name}") from e
