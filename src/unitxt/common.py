from typing import Union

from .card import TaskCard
from .collections import ItemPicker, RandomPicker
from .dataclass import OptionalField
from .operator import SourceOperator
from .recipe import Recipe, SequentialRecipe
from .schema import ToUnitxtGroup
from .splitters import RandomSampler, Sampler, SeparateSplit, SliceSplit, SpreadSplit
from .stream import MultiStream
from .templates import RenderTemplatedICL


class CommonRecipe(Recipe, SourceOperator):
    card: TaskCard
    demos_pool_name: str = "demos_pool"
    demos_taken_from: str = "train"
    demos_pool_size: int = None
    demos_field: str = "demos"
    num_demos: int = None
    sampler: Sampler = None
    instruction_item: Union[str, int] = None
    template_item: Union[str, int] = None

    def verify(self):
        super().verify()

    def prepare(self):
        steps = [
            self.card.loader,
        ]

        if self.card.preprocess_steps is not None:
            steps.extend(self.card.preprocess_steps)

        steps.append(self.card.task)

        if self.demos_pool_size is not None:
            steps.append(
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

            steps.append(
                SpreadSplit(
                    source_stream=self.demos_pool_name,
                    target_field=self.demos_field,
                    sampler=sampler,
                )
            )

        if self.card.instructions is not None:
            if not self.instruction_item is None:
                picker = ItemPicker(int(self.instruction_item))
            else:
                picker = RandomPicker()
            instruction = picker(self.card.instructions)
        else:
            instruction = None

        if self.card.templates is not None:
            if self.template_item is None:
                picker = RandomPicker()
            else:
                picker = ItemPicker(self.template_item)
            template = picker(self.card.templates)
        else:
            template = None

        render = RenderTemplatedICL(
            instruction=instruction,
            template=template,
            demos_field=self.demos_field,
        )

        steps.append(render)

        postprocessors = render.get_postprocessors()

        steps.append(
            ToUnitxtGroup(
                group="unitxt",
                metrics=self.card.task.metrics,
                postprocessors=postprocessors,
            )
        )

        self.recipe = SequentialRecipe(steps)

    def process(self) -> MultiStream:
        return self.recipe()
