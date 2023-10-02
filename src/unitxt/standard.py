from typing import List

from .card import TaskCard
from .dataclass import InternalField, OptionalField
from .formats import ICLFormat
from .instructions import Instruction
from .operator import SourceSequntialOperator, StreamingOperator
from .operators import StreamRefiner
from .recipe import Recipe
from .renderers import StandardRenderer
from .schema import ToUnitxtGroup
from .splitters import Sampler, SeparateSplit, SpreadSplit
from .templates import Template


class BaseRecipe(Recipe, SourceSequntialOperator):
    card: TaskCard
    template: Template = None
    instruction: Instruction = None
    format: ICLFormat = ICLFormat()

    max_train_instances: int = None
    max_validation_instances: int = None
    max_test_instances: int = None

    train_refiner: StreamRefiner = OptionalField(default_factory=lambda: StreamRefiner(apply_to_streams=["train"]))
    validation_refiner: StreamRefiner = OptionalField(
        default_factory=lambda: StreamRefiner(apply_to_streams=["validation"])
    )
    test_refiner: StreamRefiner = OptionalField(default_factory=lambda: StreamRefiner(apply_to_streams=["test"]))

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

        self.train_refiner.max_instances = self.max_train_instances
        self.steps.append(self.train_refiner)

        self.validation_refiner.max_instances = self.max_validation_instances
        self.steps.append(self.validation_refiner)

        self.test_refiner.max_instances = self.max_test_instances
        self.steps.append(self.test_refiner)

        postprocessors = render.get_postprocessors()

        self.steps.append(
            ToUnitxtGroup(
                group="unitxt",
                metrics=self.card.task.metrics,
                postprocessors=postprocessors,
            )
        )


class StandardRecipeWithIndexes(BaseRecipe):
    instruction_card_index: int = None
    template_card_index: int = None

    def prepare(self):
        assert (
            self.template_card_index is None or self.template is None
        ), "Specify either template or template_card_index"
        if self.template_card_index is not None:
            self.template = self.card.templates[int(self.template_card_index)]

        assert (
            self.instruction_card_index is None or self.instruction is None
        ), "Specify either instruction or instruction_card_index"
        if self.instruction_card_index is not None:
            self.instruction = self.card.instructions[int(self.instruction_card_index)]

        super().prepare()


class StandardRecipe(StandardRecipeWithIndexes):
    """
    This class represents a standard recipe for data processing and preperation.
    This class can be used to prepare a recipe
    with all necessary steps, refiners and renderers included. It allows to set various
    parameters and steps in a sequential manner for preparing the recipe.

    Attributes:
        card (TaskCard): TaskCard object associated with the recipe.
        template (Template, optional): Template object to be used for the recipe.
        instruction (Instruction, optional): Instruction object to be used for the recipe.
        format (ICLFormat, optional): ICLFormat object to be used for the recipe.
        train_refiner (StreamRefiner, optional): Train refiner to be used in the recipe.
        max_train_instances (int, optional): Maximum training instances for the refiner.
        validation_refiner (StreamRefiner, optional): Validation refiner to be used in the recipe.
        max_validation_instances (int, optional): Maximum validation instances for the refiner.
        test_refiner (StreamRefiner, optional): Test refiner to be used in the recipe.
        max_test_instances (int, optional): Maximum test instances for the refiner.
        demos_pool_size (int, optional): Size of the demos pool.
        num_demos (int, optional): Number of demos to be used.
        demos_pool_name (str, optional): Name of the demos pool. Default is "demos_pool".
        demos_taken_from (str, optional): Specifies from where the demos are taken. Default is "train".
        demos_field (str, optional): Field name for demos. Default is "demos".
        sampler (Sampler, optional): Sampler object to be used in the recipe.
        steps (List[StreamingOperator], optional): List of StreamingOperator objects to be used in the recipe.
        instruction_card_index (int, optional): Index of instruction card to be used
            for preparing the recipe.
        template_card_index (int, optional): Index of template card to be used for
            preparing the recipe.

    Methods:
        prepare(): This overridden method is used for preparing the recipe
            by arranging all the steps, refiners, and renderers in a sequential manner.

    Raises:
        AssertionError: If both template and template_card_index, or instruction and instruction_card_index
            are specified at the same time.
    """

    pass
