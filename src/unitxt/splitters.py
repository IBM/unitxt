import itertools
from abc import abstractmethod
from typing import Dict, List

from .artifact import Artifact
from .operator import InstanceOperatorWithMultiStreamAccess, MultiStreamOperator
from .random_utils import get_random
from .split_utils import (
    parse_random_mix_string,
    parse_slices_string,
    random_mix_streams,
    rename_split,
    slice_streams,
)
from .stream import MultiStream


class Splitter(MultiStreamOperator):
    pass


class RenameSplits(Splitter):
    mapper: Dict[str, str]

    def process(self, multi_stream: MultiStream) -> MultiStream:
        generators = rename_split(multi_stream, self.mapper)
        return MultiStream(generators)


class SplitRandomMix(Splitter):
    mix: Dict[str, str]

    def process(self, multi_stream: MultiStream) -> MultiStream:
        mapping = {k: parse_random_mix_string(v) for k, v in self.mix.items()}
        generators = random_mix_streams(multi_stream, mapping)
        return MultiStream.from_generators(generators)


class SeparateSplit(Splitter):
    """Separates a split (e.g. train) into several splits (e.g. train1, train2).

    sizes must indicate the size of every split except the last. If no size is give for the last split,
     it includes all the examples not allocated to any split.
    """

    from_split: str
    to_split_names: List[str]
    to_split_sizes: List[int]

    def verify(self):
        assert (
            len(self.to_split_names) == len(self.to_split_sizes)
            or len(self.to_split_names) == len(self.to_split_sizes) + 1
        ), f"Examples num should be specified to all or all but the last splits, instead given {len(self.to_split_names)} split names and {len(self.to_split_sizes)} split sizes. \n split names:{self.to_split_names} split sizes {self.to_split_sizes}"
        return super().verify()

    def process(self, multi_stream: MultiStream) -> MultiStream:
        mapping = {
            key: {key: [(None, None)]}
            for key in multi_stream.keys()
            if key != self.from_split
        }
        so_far = 0
        for name, size in itertools.zip_longest(
            self.to_split_names, self.to_split_sizes
        ):
            mapping[name] = {self.from_split: [(so_far, size)]}
            if size:
                so_far += size
        generators = slice_streams(multi_stream, mapping)
        return MultiStream.from_generators(generators)


class SliceSplit(Splitter):
    slices: Dict[str, str]

    def process(self, multi_stream: MultiStream) -> MultiStream:
        mapping = {k: parse_slices_string(v) for k, v in self.slices.items()}
        generators = slice_streams(multi_stream, mapping)
        return MultiStream.from_generators(generators)


class Sampler(Artifact):
    sample_size: int = None

    def prepare(self):
        super().prepare()
        self.set_size(self.sample_size)

    def set_size(self, size):
        if isinstance(size, str):
            assert (
                size.isdigit()
            ), f"sample_size must be a natural number, got {self.sample_size}"
            size = int(size)
        self.sample_size = size

    @abstractmethod
    def sample(
        self, instances_pool: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        pass


class RandomSampler(Sampler):
    def sample(
        self, instances_pool: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        instances_pool = list(instances_pool)
        return get_random().sample(instances_pool, self.sample_size)


class DiverseLabelsSampler(Sampler):
    choices: str = "choices"

    def prepare(self):
        super().prepare()
        self.labels = None

    def examplar_repr(self, examplar):
        if "inputs" not in examplar:
            raise ValueError(f"'inputs' field is missing from '{examplar}'.")
        inputs = examplar["inputs"]
        if self.choices not in inputs:
            raise ValueError(f"{self.choices} field is missing from '{inputs}'.")
        choices = inputs[self.choices]
        if not isinstance(choices, list):
            raise ValueError(
                f"Unexpected input choices value '{choices}'. Expected a list."
            )

        if "outputs" not in examplar:
            raise ValueError(f"'outputs' field is missing from '{examplar}'.")
        examplar_outputs = next(iter(examplar["outputs"].values()))
        if not isinstance(examplar_outputs, list):
            raise ValueError(
                f"Unexpected examplar_outputs value '{examplar_outputs}'. Expected a list."
            )

        return str([choice for choice in choices if choice in examplar_outputs])

    def divide_by_repr(self, examplars_pool):
        labels = {}
        for examplar in examplars_pool:
            label_repr = self.examplar_repr(examplar)
            if label_repr not in labels:
                labels[label_repr] = []
            labels[label_repr].append(examplar)
        return labels

    def sample(
        self, instances_pool: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        if self.labels is None:
            self.labels = self.divide_by_repr(instances_pool)
        all_labels = list(self.labels.keys())
        get_random().shuffle(all_labels)
        from collections import Counter

        total_allocated = 0
        allocations = Counter()

        while total_allocated < self.sample_size:
            for label in all_labels:
                if total_allocated < self.sample_size:
                    if len(self.labels[label]) - allocations[label] > 0:
                        allocations[label] += 1
                        total_allocated += 1
                else:
                    break

        result = []
        for label, allocation in allocations.items():
            sample = get_random().sample(self.labels[label], allocation)
            result.extend(sample)

        get_random().shuffle(result)
        return result


class SpreadSplit(InstanceOperatorWithMultiStreamAccess):
    source_stream: str = None
    target_field: str = None
    sampler: Sampler = None

    def prepare(self):
        self.local_cache = None
        self.sampler.prepare()

    def verify(self):
        assert self.source_stream is not None, "Source stream must be specified"
        assert self.target_field is not None, "Target field must be specified"
        assert self.sampler is not None, "Sampler must be specified"
        return super().verify()

    def process(
        self, instance: Dict[str, object], multi_stream: MultiStream
    ) -> Dict[str, object]:
        try:
            if self.local_cache is None:
                self.local_cache = list(multi_stream[self.source_stream])

            source_stream = self.local_cache

            sampled_instances = self.sampler.sample(source_stream)
            instance[self.target_field] = sampled_instances
            return instance
        except Exception as e:
            raise Exception(
                f"Unable to fetch instances from '{self.source_stream}' to '{self.target_field}'"
            ) from e
