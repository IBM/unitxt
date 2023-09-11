from typing import List

from .card import TaskCard
from .dataclass import InternalField
from .formats import ICLFormat
from .instructions import Instruction
from .operator import SourceSequntialOperator, StreamingOperator
from .recipe import Recipe
from .renderers import StandardRenderer
from .schema import ToUnitxtGroup
from .splitters import Sampler, SeparateSplit, SpreadSplit
from .templates import Template


class StandardRecipe(Recipe, SourceSequntialOperator):
    card: TaskCard
    template: Template
    instruction: Instruction = None
    format: ICLFormat = None

    demos_pool_size: int = None
    num_demos: int = None

    demos_pool_name: str = "demos_pool"
    demos_taken_from: str = "train"
    demos_field: str = "demos"
    sampler: Sampler = None

    steps: List[StreamingOperator] = InternalField(default_factory=list)

    def prepare(self):
        self.steps = [
            self.card.loader,
        ]

        if self.card.preprocess_steps is not None:
            self.steps.extend(self.card.preprocess_steps)

        self.steps.append(self.card.task)

        if self.demos_pool_size is not None:
            self.steps.append(
                SeparateSplit(
                    from_split=self.demos_taken_from,
                    to_split_names=[self.demos_pool_name, self.demos_taken_from],
                    to_split_sizes=[int(self.demos_pool_size)],
                )
            )

        if self.num_demos is not None:
            sampler = self.card.sampler

            if self.sampler is not None:
                sampler = self.sampler

            sampler.set_size(self.num_demos)

            self.steps.append(
                SpreadSplit(
                    source_stream=self.demos_pool_name,
                    target_field=self.demos_field,
                    sampler=sampler,
                )
            )

        render = StandardRenderer(
            instruction=self.instruction,
            template=self.template,
            format=self.format,
            demos_field=self.demos_field,
        )

        self.steps.append(render)

        postprocessors = render.get_postprocessors()

        self.steps.append(
            ToUnitxtGroup(
                group="unitxt",
                metrics=self.card.task.metrics,
                postprocessors=postprocessors,
            )
        )
