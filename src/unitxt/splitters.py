import itertools
from abc import abstractmethod
from copy import deepcopy
from random import Random
from typing import Dict, List

from .artifact import Artifact
from .operator import InstanceOperatorWithMultiStreamAccess, MultiStreamOperator
from .random_utils import new_random_generator
from .split_utils import (
    parse_random_mix_string,
    parse_slices_string,
    random_mix_streams,
    rename_split,
    slice_streams,
)
from .stream import EmptyStreamError, FaultyStreamError, MultiStream


class Splitter(MultiStreamOperator):
    pass


class RenameSplits(Splitter):
    mapper: Dict[str, str]

    def process(self, multi_stream: MultiStream) -> MultiStream:
        generators = rename_split(multi_stream, self.mapper)
        return MultiStream(generators)


class SplitRandomMix(Splitter):
    """Splits a multistream into new streams (splits), whose names, source input stream, and amount of instances, are specified by arg 'mix'.

    The keys of arg 'mix', are the names of the new streams, the values are of the form: 'name-of-source-stream[percentage-of-source-stream]'
    Each input instance, of any input stream, is selected exactly once for inclusion in any of the output streams.

    Examples:
    When processing a multistream made of two streams whose names are 'train' and 'test', by
    SplitRandomMix(mix =  { "train": "train[99%]",  "validation": "train[1%]",  "test": "test" })
    the output is a multistream, whose three streams are named 'train', 'validation', and 'test'.
    Output stream 'train' is made of randomly selected 99% of the instances of input stream 'train',
    output stream 'validation' is made of the remaining 1% instances of input 'train', and output stream 'test' is made
    of the whole of input stream 'test'.

    When processing the above input multistream by
    SplitRandomMix(mix =  { "train": "train[50%]+test[0.1]",  "validation": "train[50%]+test[0.2]",  "test": "test[0.7]" })
    the output is a multistream, whose three streams are named 'train', 'validation', and 'test'.
    Output stream 'train' is made of randomly selected 50% of the instances of input stream 'train' + randomly selected
    0.1 (i.e., 10%) of the instances of input stream 'test'.
    Output stream 'validation' is made of the remaining 50% instances of input 'train'+ randomly selected 0.2 (i.e.,
    20%) of the original instances of input 'test', that were not selected for output 'train',
    and output stream 'test' is made of the remaining instances of input 'test'.
    """

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
    remove_targets_from_source_split: bool = True

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
            if not self.remove_targets_from_source_split or key != self.from_split
        }
        so_far = 0
        for name, size in itertools.zip_longest(
            self.to_split_names, self.to_split_sizes
        ):
            if self.remove_targets_from_source_split or name != self.from_split:
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
    random_generator: Random = new_random_generator(sub_seed="Sampler")

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

    def init_new_random_generator(self):
        self.random_generator = new_random_generator(
            sub_seed="init_new_random_generator"
        )

    @abstractmethod
    def sample(
        self, instances_pool: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        pass

    def filter_source_by_instance(
        self, instances_pool: List[Dict[str, object]], instance: Dict[str, object]
    ) -> List[Dict[str, object]]:
        if "inputs" not in instance:
            raise ValueError(f"'inputs' field is missing from '{instance}'.")
        # l = list(filter(lambda x: x["inputs"] != instance["inputs"], instances_pool))
        try:
            return [
                item for item in instances_pool if item["inputs"] != instance["inputs"]
            ]
        except Exception as e:
            raise e


class RandomSampler(Sampler):
    def sample(
        self, instances_pool: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        instances_pool = list(instances_pool)
        return self.random_generator.sample(instances_pool, self.sample_size)


class DiverseLabelsSampler(Sampler):
    """Selects a balanced sample of instances based on an output field.

    (used for selecting demonstrations in-context learning)

    The field must contain list of values e.g ['dog'], ['cat'], ['dog','cat','cow'].
    The balancing is done such that each value or combination of values
    appears as equals as possible in the samples.

    The `choices` param is required and determines which values should be considered.

    Example:
        If choices is ['dog,'cat'] , then the following combinations will be considered.
        ['']
        ['cat']
        ['dog']
        ['dog','cat']

        If the instance contains a value not in the 'choice' param, it is ignored. For example,
        if choices is ['dog,'cat'] and the instance field is ['dog','cat','cow'], then 'cow' is ignored
        then the instance is considered as ['dog','cat'].

    Args:
        sample_size - number of samples to extract
        choices - name of input field that contains the list of values to balance on
        labels - name of output field with labels that must be balanced


    """

    choices: str = "choices"
    labels: str = "labels"
    include_empty_label: bool = True

    def prepare(self):
        super().prepare()
        self.labels_cache = None

    def exemplar_repr(self, exemplar):
        if "inputs" not in exemplar:
            raise ValueError(f"'inputs' field is missing from '{exemplar}'.")
        inputs = exemplar["inputs"]
        if self.choices not in inputs:
            raise ValueError(f"'{self.choices}' field is missing from '{inputs}'.")
        choices = inputs[self.choices]
        if not isinstance(choices, list):
            if isinstance(choices, str):
                choices = [choices]
            else:
                raise ValueError(
                    f"Unexpected input choices value '{choices}'. Expected a list or a string."
                )

        if "outputs" not in exemplar:
            raise ValueError(f"'outputs' field is missing from '{exemplar}'.")
        outputs = exemplar["outputs"]
        if self.labels not in outputs:
            raise ValueError(f"'{self.labels}' field is missing from '{outputs}'.")

        exemplar_outputs = exemplar["outputs"][self.labels]
        if not isinstance(exemplar_outputs, list):
            raise ValueError(
                f"Unexpected exemplar_outputs value '{exemplar_outputs}'. Expected a list."
            )

        return str([choice for choice in choices if choice in exemplar_outputs])

    def divide_by_repr(self, exemplars_pool):
        labels = {}
        for exemplar in exemplars_pool:
            label_repr = self.exemplar_repr(exemplar)
            if label_repr == "[]" and not self.include_empty_label:
                continue
            if label_repr not in labels:
                labels[label_repr] = []
            labels[label_repr].append(exemplar)
        return labels

    def sample(
        self, instances_pool: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        if self.labels_cache is None:
            self.labels_cache = self.divide_by_repr(instances_pool)
        all_labels = list(self.labels_cache.keys())
        self.random_generator.shuffle(all_labels)
        from collections import Counter

        if self.sample_size > len(instances_pool):
            raise ValueError(
                f"Request sample size {self.sample_size} is greater than number of instances {len(instances_pool)}"
            )
        total_allocated = 0
        allocations = Counter()

        while total_allocated < self.sample_size:
            for label in all_labels:
                if total_allocated < self.sample_size:
                    if len(self.labels_cache[label]) - allocations[label] > 0:
                        allocations[label] += 1
                        total_allocated += 1
                else:
                    break

        result = []
        for label, allocation in allocations.items():
            sample = self.random_generator.sample(self.labels_cache[label], allocation)
            result.extend(sample)

        self.random_generator.shuffle(result)
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
                self.local_cache = deepcopy(list(multi_stream[self.source_stream]))

            source_stream = self.local_cache
            source_stream = self.sampler.filter_source_by_instance(
                source_stream, instance
            )
            if len(source_stream) < self.sampler.sample_size:
                raise ValueError(
                    f"Size of population to sample from: {len(source_stream)} is smaller than the needed sample_size: {self.sampler.sample_size}."
                )
            sampled_instances = self.sampler.sample(source_stream)
            instance[self.target_field] = sampled_instances
            return instance
        except FaultyStreamError as e:
            raise EmptyStreamError(
                f"Unable to fetch instances from '{self.source_stream}' to '{self.target_field}', due to {e.__class__.__name__}: {e}"
            ) from e
