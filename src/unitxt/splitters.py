from dataclasses import field
from typing import Dict, List, Optional

from .artifact import Artifact
from .generator_utils import ReusableGenerator
from .operator import InstanceOperatorWithGlobalAccess, MultiStreamOperator
from .stream import MultiStream


class Splitter(MultiStreamOperator):
    pass


import random

from .split_utils import (
    parse_random_mix_string,
    parse_slices_string,
    random_mix_streams,
    slice_streams,
)


class SplitRandomMix(Splitter):
    mix: Dict[str, str]

    def process(self, multi_stream: MultiStream) -> MultiStream:
        mapping = {k: parse_random_mix_string(v) for k, v in self.mix.items()}
        generators = random_mix_streams(multi_stream, mapping)
        return MultiStream.from_generators(generators, streaming=True)



class SliceSplit(Splitter):
    slices: Dict[str, str]

    def process(self, multi_stream: MultiStream) -> MultiStream:
        mapping = {k: parse_slices_string(v) for k, v in self.slices.items()}
        generators = slice_streams(multi_stream, mapping)
        return MultiStream.from_generators(generators, streaming=True)


class Sampler(Artifact):
    sample_size: int


class RandomSampler(Sampler):
    def sample(self, instances_pool: List[Dict[str, object]]) -> List[Dict[str, object]]:
        instances_pool = list(instances_pool)
        return random.sample(instances_pool, self.sample_size)


class SpreadSplit(InstanceOperatorWithGlobalAccess):
    source_stream: str = None
    target_field: str = None
    sampler: Sampler = None

    def prepare(self):
        self.accessible_streams = [self.source_stream]
        self.cache_accessible_streams = True
        self.local_cache = None

    def verify(self):
        assert self.source_stream is not None, "Source stream must be specified"
        assert self.target_field is not None, "Target field must be specified"
        assert self.sampler is not None, "Sampler must be specified"
        return super().verify()

    def process(self, instance: Dict[str, object], multi_stream: MultiStream) -> Dict[str, object]:
        if self.local_cache is None:
            self.local_cache = list(multi_stream[self.source_stream])

        source_stream = self.local_cache

        sampled_instances = self.sampler.sample(source_stream)
        instance[self.target_field] = sampled_instances
        return instance


if __name__ == "__main__":
    # some tests
    import random

    random.seed(0)
    splitter = SplitRandomMix(
        mix={
            "train": "train[90%]+validation[50%]",
            "validation": "train[10%]+validation[50%]",
            "test": "test",
        }
    )


    def generator(name, size):
        for i in range(size):
            yield {"text": f"{name}_{i}"}


    stream = MultiStream.from_generators(
        {
            "train": ReusableGenerator(generator, gen_kwargs={"name": "train", "size": 10}),
            "validation": ReusableGenerator(generator, gen_kwargs={"name": "validation", "size": 10}),
            "test": ReusableGenerator(generator, gen_kwargs={"name": "test", "size": 10}),
        }
    )

    ds = splitter(stream)
    for key, value in ds.items():
        print(key)
        for item in value:
            print(item)

    splitter = SliceSplit(
        slices={
            "train": "train[:2]+train[2:4]",
            "validation": "train[4:6]",
            "test": "train[6:]+test",
        }
    )

    ds = splitter(stream)
    for key, value in ds.items():
        print(key)
        for item in value:
            print(item)
