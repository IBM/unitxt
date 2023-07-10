from .stream import MultiStream
from .operator import SourceOperator
from .card import TaskCard
from .splitters import SliceSplit, SpreadSplit, RandomSampler
from .recipe import SequentialRecipe, Recipe
from .collections import ItemPicker, RandomPicker
from .templates import RenderTemplatedICL
from .schema import ToUnitxtGroup

from typing import Union


class CommonRecipe(Recipe, SourceOperator):
    card: TaskCard
    demos_pool_name: str = "demos_pool"
    demos_pool_size: int = None
    demos_field: str = "demos"
    num_demos: int = None
    sampler_type: str = "random"
    instruction_item: Union[str, int] = None
    template_item: Union[str, int] = None

    def verify(self):
        self.sampler_type in ["random"]

    def prepare(self):
        steps = [
            self.card.loader,
        ]

        if self.card.preprocess_steps is not None:
            steps.extend(self.card.preprocess_steps)

        steps.append(self.card.task)

        if self.demos_pool_size is not None:
            steps.append(
                SliceSplit(
                    slices={
                        self.demos_pool_name: f"train[:{self.demos_pool_size}]",
                        "train": f"train[{self.demos_pool_size}:]",
                        "validation": "validation",
                        "test": "test",
                    }
                )
            )

        if self.num_demos is not None:
            if self.sampler_type == "random":
                sampler = RandomSampler(sample_size=self.num_demos)

            steps.append(
                SpreadSplit(
                    source_stream=self.demos_pool_name,
                    target_field=self.demos_field,
                    sampler=sampler,
                )
            )

        if self.card.instructions is not None:
            if self.instruction_item is None:
                picker = ItemPicker(self.instruction_item)
            else:
                picker = RandomPicker()
            instruction = picker(self.card.instructions)
        else:
            instruction = None

        if self.card.templates is not None:
            if self.template_item is None:
                picker = ItemPicker(self.template_item)
            else:
                picker = RandomPicker()
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
