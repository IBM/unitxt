from typing import List

from .card import TaskCard
from .dataclass import Field, InternalField, OptionalField
from .formats import Format, SystemFormat
from .instructions import EmptyInstruction, Instruction
from .logging_utils import get_logger
from .operator import SourceSequentialOperator, StreamingOperator
from .operators import (
    Augmentor,
    NullAugmentor,
    StreamRefiner,
)
from .recipe import Recipe
from .schema import ToUnitxtGroup
from .splitters import Sampler, SeparateSplit, SpreadSplit
from .templates import Template

logger = get_logger()


# Used to give meaningful name to recipe steps
class CreateDemosPool(SeparateSplit):
    pass


class AddDemosField(SpreadSplit):
    pass


class BaseRecipe(Recipe, SourceSequentialOperator):
    card: TaskCard
    template: Template = None
    instruction: Instruction = Field(default_factory=EmptyInstruction)
    format: Format = Field(default_factory=SystemFormat)

    loader_limit: int = None

    max_train_instances: int = None
    max_validation_instances: int = None
    max_test_instances: int = None

    train_refiner: StreamRefiner = OptionalField(default_factory=StreamRefiner)
    validation_refiner: StreamRefiner = OptionalField(default_factory=StreamRefiner)
    test_refiner: StreamRefiner = OptionalField(default_factory=StreamRefiner)

    demos_pool_size: int = None
    num_demos: int = 0

    demos_pool_name: str = "demos_pool"
    demos_taken_from: str = "train"
    demos_field: str = "demos"
    sampler: Sampler = None

    augmentor: Augmentor = OptionalField(default_factory=NullAugmentor)

    steps: List[StreamingOperator] = InternalField(default_factory=list)

    def before_process_multi_stream(self):
        super().before_process_multi_stream()
        if self.sampler:  # e.g. when num_demos is 0, the sampler may not be initialized
            self.sampler.init_new_random_generator()

    def verify(self):
        super().verify()
        if self.num_demos > 0:
            if self.demos_pool_size is None or self.demos_pool_size < 1:
                raise ValueError(
                    "When using demonstrations both num_demos and demos_pool_size should be assigned with postive integers."
                )
            if self.demos_pool_size < self.num_demos:
                raise ValueError(
                    f"num_demos (got: {self.num_demos}) should not exceed demos_pool_size (got: {self.demos_pool_size})"
                )
            if self.loader_limit and self.demos_pool_size > self.loader_limit:
                raise ValueError(
                    f"demos_pool_size should not exceed loader_limit ({self.loader_limit}), Got demos_pool_size={self.demos_pool_size}"
                )

        if self.loader_limit:
            if self.max_test_instances and self.max_test_instances > self.loader_limit:
                raise ValueError(
                    f"max_test_instances should not exceed loader_limit ({self.loader_limit}), Got max_test_instances={self.max_test_instances}"
                )
            if (
                self.max_validation_instances
                and self.max_validation_instances > self.loader_limit
            ):
                raise ValueError(
                    f"max_validation_instances should not exceed loader_limit ({self.loader_limit}), Got max_validation_instances={self.max_validation_instances}"
                )
            if (
                self.max_train_instances
                and self.max_train_instances > self.loader_limit
            ):
                raise ValueError(
                    f"max_train_instances should not exceed loader_limit ({self.loader_limit}), Got max_train_instances={self.max_train_instances}"
                )

    def prepare(self):
        self.steps = [
            self.card.loader,
        ]

        if self.loader_limit:
            self.card.loader.loader_limit = self.loader_limit
            logger.info(f"Loader line limit was set to  {self.loader_limit}")
            self.steps.append(StreamRefiner(max_instances=self.loader_limit))

        if self.card.preprocess_steps is not None:
            self.steps.extend(self.card.preprocess_steps)

        self.steps.append(self.card.task)

        if self.augmentor.augment_task_input:
            self.augmentor.set_task_input_fields(self.card.task.augmentable_inputs)
            self.steps.append(self.augmentor)

        if self.demos_pool_size is not None:
            self.steps.append(
                CreateDemosPool(
                    from_split=self.demos_taken_from,
                    to_split_names=[self.demos_pool_name, self.demos_taken_from],
                    to_split_sizes=[int(self.demos_pool_size)],
                )
            )

        if self.num_demos > 0:
            if self.sampler is None:
                if self.card.sampler is None:
                    raise ValueError(
                        "Unexpected None value for card.sampler. "
                        "To use num_demos > 0, please set a sampler on the TaskCard."
                    )
                self.sampler = self.card.sampler

            self.sampler.set_size(self.num_demos)

        self.train_refiner.max_instances = self.max_train_instances
        self.train_refiner.apply_to_streams = ["train"]
        self.steps.append(self.train_refiner)

        self.validation_refiner.max_instances = self.max_validation_instances
        self.validation_refiner.apply_to_streams = ["validation"]
        self.steps.append(self.validation_refiner)

        self.test_refiner.max_instances = self.max_test_instances
        self.test_refiner.apply_to_streams = ["test"]
        self.steps.append(self.test_refiner)

        self.steps.append(self.template)
        if self.num_demos > 0:
            self.steps.append(
                AddDemosField(
                    source_stream=self.demos_pool_name,
                    target_field=self.demos_field,
                    sampler=self.sampler,
                )
            )
        self.steps.append(self.instruction)
        self.steps.append(self.format)
        if self.augmentor.augment_model_input:
            self.steps.append(self.augmentor)

        postprocessors = self.template.get_postprocessors()

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
        ), f"Specify either template ({self.template}) or template_card_index ({self.template_card_index}) but not both"
        assert not (
            self.template_card_index is None and self.template is None
        ), "Specify either template or template_card_index in card"
        if self.template_card_index is not None:
            try:
                self.template = self.card.templates[self.template_card_index]
            except Exception as e:
                if isinstance(self.card.templates, dict):
                    options = self.card.templates.keys()
                else:
                    options = list(range(0, len(self.card.templates)))
                raise ValueError(
                    f"card_template_index '{self.template_card_index}' is not in card. Available options: {options}"
                ) from e
        assert (
            self.instruction_card_index is None or self.instruction is None
        ), "Specify either instruction or instruction_card_index"
        if self.instruction_card_index is not None:
            self.instruction = self.card.instructions[int(self.instruction_card_index)]

        super().prepare()


class StandardRecipe(StandardRecipeWithIndexes):
    """This class represents a standard recipe for data processing and preparation.

    This class can be used to prepare a recipe.
    with all necessary steps, refiners and renderers included. It allows to set various
    parameters and steps in a sequential manner for preparing the recipe.

    Attributes:
        card (TaskCard): TaskCard object associated with the recipe.
        template (Template, optional): Template object to be used for the recipe.
        instruction (Instruction, optional): Instruction object to be used for the recipe.
        loader_limit (int, optional): Specifies the maximum number of instances per stream to be returned from the loader (used to reduce loading time in large datasets)
        format (SystemFormat, optional): SystemFormat object to be used for the recipe.
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
        augmentor (Augmentor) : Augmentor to be used to pseudo randomly augment the source text
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
