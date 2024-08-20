from typing import Dict, Union

from .dataclass import NonPositionalField
from .formats import Format
from .fusion import FixedFusion, WeightedFusion
from .operator import SourceOperator
from .standard import StandardRecipe
from .stream import MultiStream
from .system_prompts import SystemPrompt


class BaseBenchmark(SourceOperator):
    format: Format = NonPositionalField(default=None)
    num_demos: int = NonPositionalField(default=None)
    system_prompt: SystemPrompt = NonPositionalField(default=None)
    loader_limit: int = NonPositionalField(default=None)


class Benchmark(BaseBenchmark):
    subsets: Dict[str, Union[StandardRecipe, BaseBenchmark]]

    max_total_samples: int = None
    max_samples_per_subset: int = None

    def verify(self):
        if (
            self.max_total_samples is not None
            and self.max_samples_per_subset is not None
        ):
            raise ValueError("Set either max_total_samples or max_samples_per_subset")

    def prepare(self):
        if self.format is not None or self.num_demos is not None:
            for subset in self.subsets.values():
                if self.num_demos is not None:
                    subset.num_demos = self.num_demos
                if self.format is not None:
                    subset.format = self.format
                if self.system_prompt is not None:
                    subset.system_prompt = self.system_prompt
                if self.loader_limit is not None:
                    subset.loader_limit = self.loader_limit
                subset.prepare()

    def process(
        self,
    ) -> MultiStream:
        if self.max_total_samples is None:
            operator = FixedFusion(
                subsets=self.subsets,
                max_instances_per_subset=self.max_samples_per_subset,
            )
        else:
            operator = WeightedFusion(
                subsets=self.subsets, max_total_samples=self.max_total_samples
            )

        return operator()
