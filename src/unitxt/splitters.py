import itertools
from abc import abstractmethod
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

from .artifact import Artifact
from .dict_utils import dict_get
from .operator import InstanceOperator, MultiStreamOperator
from .random_utils import new_random_generator
from .split_utils import (
    parse_random_mix_string,
    parse_slices_string,
    random_mix_streams,
    rename_split,
    slice_streams,
)
from .stream import MultiStream
from .type_utils import isoftype
from .utils import recursive_copy


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


def get_random_generator_based_on_instance(instance, local_seed=None):
    sub_seed = {**instance["input_fields"]}
    if local_seed is not None:
        sub_seed["local_seed"] = local_seed
    return new_random_generator(sub_seed=sub_seed)


class Sampler(Artifact):
    @abstractmethod
    def sample(
        self,
        sample_size: int,
        instances_pool: List[Dict[str, Any]],
        instance: Dict[str, Any],
        sampling_seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        pass

    def filter_source_by_instance(
        self, instances_pool: List[Dict[str, Any]], instance: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if "input_fields" not in instance:
            raise ValueError(f"'input_fields' field is missing from '{instance}'.")
        try:
            return [
                item
                for item in instances_pool
                if item["input_fields"] != instance["input_fields"]
            ]
        except Exception as e:
            raise e


class RandomSampler(Sampler):
    """Selects a random sample of instances."""

    def sample(
        self,
        sample_size,
        instances_pool: List[Dict[str, object]],
        instance: Optional[Dict[str, object]],
        sampling_seed: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        instances_pool = list(instances_pool)
        random_generator = get_random_generator_based_on_instance(
            instance, local_seed=sampling_seed
        )
        return random_generator.sample(instances_pool, sample_size)


class FixedIndicesSampler(Sampler):
    """Selects a fix set of samples based on a list of indices."""

    indices: List[int]

    def verify(self):
        assert isoftype(
            self.indices, List[int]
        ), f"'indices' of {self.__class__.__name__} must be List[int]. Value {self.indices} is of type {type(self.indices)}"
        super().verify()

    def sample(
        self,
        sample_size,
        instances_pool: List[Dict[str, object]],
        instance: Optional[Dict[str, object]],
        sampling_seed: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        num_instances = len(instances_pool)

        instances = []
        for index in self.indices[0:sample_size]:
            if index >= num_instances:
                raise ValueError(
                    f"FixedIndicesSampler 'indices' field contains index ({index}) which is out of bounds of the instance pool ( of size {num_instances})"
                )
            instances.append(instances_pool[index])
        return instances


class CloseTextSampler(Sampler):
    """Selects the samples of instances which are the closest textual match to the given instance.

    Comparison is done based on a given field in the instance.

    """

    field: str

    def sample(
        self,
        sample_size: int,
        instances_pool: List[Dict[str, object]],
        instance: Dict[str, object],
        sampling_seed: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        field = f"input_fields/{self.field}"
        value = dict_get(instance, field)

        instances_pool = list(instances_pool)

        # Get 'sample_size'  closest matchest texts based on field
        options = []
        for instance_in_pool in instances_pool:
            options.append(dict_get(instance_in_pool, field))
        closest_matches = get_close_matches(value, options, n=sample_size, cutoff=0)
        # Randmly select 'sample_size' instances that are from the closest matches text
        # (There may be multiple instance with same text in the given field, and the order returned is
        # is also randomized )
        instances_pool = [
            instance_in_pool
            for instance_in_pool in instances_pool
            if dict_get(instance_in_pool, field) in closest_matches
        ]
        random_generator = get_random_generator_based_on_instance(instance)
        return random_generator.sample(instances_pool, sample_size)


class DiverseLabelsSampler(Sampler):
    """Selects a balanced sample of instances based on an output field.

    (used for selecting demonstrations in-context learning)

    The field must contain list of values e.g ['dog'], ['cat'], ['dog','cat','cow'].
    The balancing is done such that each value or combination of values
    appears as equals as possible in the samples.

    The `choices` param is required and determines which values should be considered.

    Example:
        If choices is ['dog','cat'] , then the following combinations will be considered.
        ['']
        ['cat']
        ['dog']
        ['dog','cat']

        If the instance contains a value not in the 'choice' param, it is ignored. For example,
        if choices is ['dog','cat'] and the instance field is ['dog','cat','cow'], then 'cow' is ignored
        then the instance is considered as ['dog','cat'].

    Args:
        sample_size (int):
            number of samples to extract
        choices (str):
            name of input field that contains the list of values to balance on
        labels (str):
            name of output field with labels that must be balanced

    """

    choices: str = "choices"
    labels: str = "labels"
    include_empty_label: bool = True

    def prepare(self):
        super().prepare()
        self.labels_cache = None

    def exemplar_repr(self, exemplar):
        if "input_fields" not in exemplar:
            raise ValueError(f"'input_fields' field is missing from '{exemplar}'.")
        inputs = exemplar["input_fields"]
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

        if "reference_fields" not in exemplar:
            raise ValueError(f"'reference_fields' field is missing from '{exemplar}'.")
        outputs = exemplar["reference_fields"]
        if self.labels not in outputs:
            raise ValueError(f"'{self.labels}' field is missing from '{outputs}'.")

        exemplar_outputs = exemplar["reference_fields"][self.labels]
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
        self,
        sample_size: int,
        instances_pool: List[Dict[str, object]],
        instance: Optional[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        if self.labels_cache is None:
            self.labels_cache = self.divide_by_repr(instances_pool)
        all_labels = list(self.labels_cache.keys())
        random_generator = get_random_generator_based_on_instance(instance)
        random_generator.shuffle(all_labels)
        from collections import Counter

        if sample_size > len(instances_pool):
            raise ValueError(
                f"Request sample size {sample_size} is greater than number of instances {len(instances_pool)}"
            )
        total_allocated = 0
        allocations = Counter()

        while total_allocated < sample_size:
            for label in all_labels:
                if total_allocated < sample_size:
                    if len(self.labels_cache[label]) - allocations[label] > 0:
                        allocations[label] += 1
                        total_allocated += 1
                else:
                    break

        result = []
        for label, allocation in allocations.items():
            sample = random_generator.sample(self.labels_cache[label], allocation)
            result.extend(sample)

        random_generator.shuffle(result)
        return result


class AssignDemosToInstance(InstanceOperator):
    from_field: str
    to_field: str
    sampler: Sampler
    skip_demoed_instances: bool = False
    sampling_seed: Optional[int] = None

    def prepare(self):
        self.local_cache = None
        self.sampler.prepare()

    @abstractmethod
    def get_sample_size(self, instance) -> int:
        pass

    def process(
        self, instance: Dict[str, Any], multi_stream: MultiStream
    ) -> Dict[str, Any]:
        if self.skip_demoed_instances and self.to_field in instance:
            if self.from_field in instance:
                instance.pop(self.from_field)
            return instance

        demos_pool = instance[self.from_field]
        sample_size = self.get_sample_size(instance)
        source_stream = self.sampler.filter_source_by_instance(demos_pool, instance)
        if len(source_stream) < sample_size:
            raise ValueError(
                f"Size of population to sample from: {len(source_stream)} is smaller than the needed sample_size: {sample_size}. Please consider increasing increasing the demos pool, for which you may need to increase loader_limit or employ a less strict stream filtering."
            )
        sampled_instances = self.sampler.sample(
            sample_size=sample_size,
            instances_pool=source_stream,
            instance=instance,
            sampling_seed=self.sampling_seed,
        )
        instance[self.to_field] = recursive_copy(sampled_instances)
        instance.pop(self.from_field)  # pop the field pointing to the demos_pool
        return instance


class ConstantSizeSample(AssignDemosToInstance):
    sample_size: int

    def get_sample_size(self, instance) -> int:
        return self.sample_size


class RandomSizeSample(AssignDemosToInstance):
    sample_sizes: List[int]

    def get_sample_size(self, instance) -> int:
        random_generator = get_random_generator_based_on_instance(instance)
        return random_generator.choice(self.sample_sizes)
