from abc import abstractmethod
from typing import Dict, List, Optional, Union

from .dataclass import NonPositionalField
from .formats import Format
from .fusion import FixedFusion
from .operator import SourceOperator
from .standard import DatasetRecipe
from .stream import MultiStream
from .system_prompts import SystemPrompt


class BaseBenchmark(SourceOperator):
    format: Format = NonPositionalField(default=None)
    num_demos: int = NonPositionalField(default=None)
    system_prompt: SystemPrompt = NonPositionalField(default=None)
    loader_limit: int = NonPositionalField(default=None)
    splits: List[str] = NonPositionalField(
        default_factory=lambda: ["train", "validation", "test"]
    )
    subset: Optional[str] = NonPositionalField(default=None)

    @abstractmethod
    def reset(self):
        pass


class Benchmark(BaseBenchmark):
    subsets: Dict[str, Union[DatasetRecipe, BaseBenchmark]]

    max_total_samples: int = None
    max_samples_per_subset: int = None
    max_train_instances: int = None
    max_validation_instances: int = None
    max_test_instances: int = None

    def verify(self):
        super().verify()
        if (
            self.max_total_samples is not None
            and self.max_samples_per_subset is not None
        ):
            raise ValueError("Set either max_total_samples or max_samples_per_subset")

    def prepare_args(self):
        self.subsets = dict(self.subsets)

    def reset(self):
        if (
            self.format is not None
            or self.num_demos is not None
            or self.system_prompt is not None
            or self.loader_limit is not None
        ):
            for subset in self.subsets.values():
                if self.num_demos is not None:
                    subset.num_demos = self.num_demos
                if self.format is not None:
                    subset.format = self.format
                if self.system_prompt is not None:
                    subset.system_prompt = self.system_prompt
                if self.loader_limit is not None:
                    subset.loader_limit = self.loader_limit

                subset.reset()

    def prepare(self):
        super().prepare()

        self.reset()

    def process(
        self,
    ) -> MultiStream:
        if self.subset is not None:
            subsets = {self.subset: self.subsets[self.subset]}
        else:
            subsets = self.subsets

        max_instances_per_split = {}
        if self.max_train_instances is not None:
            max_instances_per_split["train"] = self.max_train_instances
        if self.max_validation_instances is not None:
            max_instances_per_split["validation"] = self.max_validation_instances
        if self.max_test_instances is not None:
            max_instances_per_split["test"] = self.max_test_instances
        if len(max_instances_per_split) == 0:
            max_instances_per_split = None

        if self.max_total_samples is None:
            operator = FixedFusion(
                subsets=subsets,
                max_instances_per_subset=self.max_samples_per_subset,
                max_instances_per_split=max_instances_per_split,
                include_splits=self.splits,
            )
        else:
            raise NotImplementedError()

        return operator()
