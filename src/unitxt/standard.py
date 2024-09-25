from typing import List, Optional, Union

from .augmentors import (
    Augmentor,
    FinalStateInputsAugmentor,
    NullAugmentor,
    TaskInputsAugmentor,
)
from .card import TaskCard
from .collections_operators import GetLength
from .dataclass import Field, InternalField, NonPositionalField, OptionalField
from .formats import Format, SystemFormat
from .logging_utils import get_logger
from .operator import SequentialOperator, SourceSequentialOperator, StreamingOperator
from .operators import Set, StreamRefiner
from .recipe import Recipe
from .schema import Finalize
from .serializers import SingleTypeSerializer
from .settings_utils import get_constants
from .splitters import ConstantSizeSample, RandomSizeSample, Sampler, SeparateSplit
from .stream import MultiStream
from .system_prompts import EmptySystemPrompt, SystemPrompt
from .task import Task
from .templates import ApplyRandomTemplate, ApplySingleTemplate, Template, TemplatesList

constants = get_constants()
logger = get_logger()


# Used to give meaningful name to recipe steps
class CreateDemosPool(SeparateSplit):
    pass


class BaseRecipe(Recipe, SourceSequentialOperator):
    # Base parameters
    card: TaskCard = None
    task: Task = None
    template: Union[Template, List[Template], TemplatesList] = None
    system_prompt: SystemPrompt = Field(default_factory=EmptySystemPrompt)
    format: Format = Field(default_factory=SystemFormat)
    serializer: Union[SingleTypeSerializer, List[SingleTypeSerializer]] = None

    # Additional parameters
    template_card_index: int = NonPositionalField(default=None)
    metrics: List[str] = NonPositionalField(default=None)
    postprocessors: List[str] = NonPositionalField(default=None)

    group_by: List[Union[str, List[str]]] = []

    loader_limit: int = None

    max_train_instances: int = None
    max_validation_instances: int = None
    max_test_instances: int = None

    train_refiner: StreamRefiner = OptionalField(default_factory=StreamRefiner)
    validation_refiner: StreamRefiner = OptionalField(default_factory=StreamRefiner)
    test_refiner: StreamRefiner = OptionalField(default_factory=StreamRefiner)

    demos_pool_size: int = None
    num_demos: Optional[Union[int, List[int]]] = 0
    demos_removed_from_data: bool = True

    demos_pool_name: str = "demos_pool"
    demos_taken_from: str = "train"
    demos_field: str = "demos"
    sampler: Sampler = None

    augmentor: Augmentor = OptionalField(default_factory=NullAugmentor)

    steps: List[StreamingOperator] = InternalField(default_factory=list)

    def before_process_multi_stream(self):
        super().before_process_multi_stream()

    @property
    def max_demos_size(self):
        if isinstance(self.num_demos, list):
            return max(self.num_demos)
        return self.num_demos

    def verify(self):
        super().verify()

        if self.task is None and self.card is None:
            raise ValueError("Set card or task in the recipe")

        if self.card is None and (
            self.num_demos > 0 or self.demos_pool_size is not None
        ):
            raise ValueError(
                "To use num_demos and demos_pool_size in recipe set a card."
            )

        if self.use_demos:
            if self.demos_pool_size is None or self.demos_pool_size < 1:
                raise ValueError(
                    "When using demonstrations both num_demos and demos_pool_size should be assigned with positive integers."
                )
            if self.demos_pool_size < self.max_demos_size:
                raise ValueError(
                    f"num_demos (got: {self.max_demos_size}) should not exceed demos_pool_size (got: {self.demos_pool_size})"
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
        if self.metrics is not None and not isinstance(self.metrics, List):
            raise ValueError(
                f"metrics must be a list of metrics.  Got metrics = {self.metrics}"
            )
        if self.postprocessors is not None and not isinstance(
            self.postprocessors, List
        ):
            raise ValueError(
                f"post processors must be a list of post processor.  Got postprocessors = {self.postprocessors}"
            )

        if self.template is None:
            raise ValueError(
                "You must set in the recipe either `template`, `template_card_index` or `templates`."
            )

        if isinstance(self.template, list):
            for template in self.template:
                self.verify_template(template)
        else:
            self.verify_template(self.template)

        if self.serializer is not None:
            if not isinstance(self.serializer, list):
                self.serializer = [self.serializer]
            self.template.serializer.add_serializers(self.serializer)

    def prepare_refiners(self):
        self.train_refiner.max_instances = self.max_train_instances
        self.train_refiner.apply_to_streams = ["train"]
        self.processing.steps.append(self.train_refiner)

        self.validation_refiner.max_instances = self.max_validation_instances
        self.validation_refiner.apply_to_streams = ["validation"]
        self.processing.steps.append(self.validation_refiner)

        self.test_refiner.max_instances = self.max_test_instances
        self.test_refiner.apply_to_streams = ["test"]
        self.processing.steps.append(self.test_refiner)

    def verify_template(self, template):
        if not isinstance(template, Template):
            raise ValueError(
                f"template argument must be an object of type Template. Got template = {template}"
            )

    def set_pipelines(self):
        self.loading = SequentialOperator(
            __description__="Loading the data from the data source."
        )
        self.metadata = SequentialOperator(
            __description__="Adding metadata (e.g. format, system prompt, template)  "
        )
        self.standardization = SequentialOperator(
            __description__="Standardizing the raw dataset fields to task field definition."
        )

        self.processing = SequentialOperator(
            __description__="Setting task fields (and selecting demos per sample if needed)."
        )
        self.verbalization = SequentialOperator()
        self.verbalization.__description__ = "Verbalizing the input to the model and gold references to the 'source', 'target' and 'references' fields."
        self.finalize = SequentialOperator()
        self.finalize.__description__ = "Adding post processors. Removing intermediate fields. Creating the final output dataset."

        self.steps = [
            self.loading,
            self.metadata,
            self.standardization,
            self.processing,
            self.metadata,
            self.verbalization,
            self.finalize,
        ]

        self.inference_instance = SequentialOperator()

        self.inference_instance.steps = [
            self.metadata,
            self.processing,
            self.metadata,
        ]

        self.inference_demos = SourceSequentialOperator()

        self.inference_demos.steps = [
            self.loading,
            self.metadata,
            self.standardization,
            self.processing,
            self.metadata,
        ]

        self.inference = SequentialOperator()

        self.inference.steps = [self.verbalization, self.finalize]

        self._demos_pool_cache = None

    def production_preprocess(self, task_instances):
        ms = MultiStream.from_iterables({constants.inference_stream: task_instances})
        return list(self.inference_instance(ms)[constants.inference_stream])

    def production_demos_pool(self):
        if self.use_demos:
            if self._demos_pool_cache is None:
                self._demos_pool_cache = list(
                    self.inference_demos()[self.demos_pool_name]
                )
            return self._demos_pool_cache
        return []

    @property
    def has_custom_demos_pool(self):
        return self.demos_pool_size is not None and self.demos_pool_size > 0

    @property
    def use_demos(self):
        return self.num_demos is not None and self.max_demos_size > 0

    def produce(self, task_instances):
        """Use the recipe in production to produce model ready query from standard task instance."""
        self.before_process_multi_stream()
        multi_stream = MultiStream.from_iterables(
            {
                constants.inference_stream: self.production_preprocess(task_instances),
                self.demos_pool_name: self.production_demos_pool(),
            }
        )
        multi_stream = self.inference(multi_stream)
        return list(multi_stream[constants.inference_stream])

    def reset_pipeline(self):
        if self.card and self.card.preprocess_steps is None:
            self.card.preprocess_steps = []

        if self.task is None:
            self.task = self.card.task

        self.set_pipelines()

        if self.card is not None:
            loader = self.card.loader
            if self.loader_limit:
                loader.loader_limit = self.loader_limit
                logger.info(f"Loader line limit was set to  {self.loader_limit}")
            self.loading.steps.append(loader)

            # This is required in case loader_limit is not enforced by the loader
            if self.loader_limit:
                self.loading.steps.append(
                    StreamRefiner(max_instances=self.loader_limit)
                )

        self.metadata.steps.append(
            Set(
                fields={
                    "recipe_metadata/system_prompt": self.system_prompt,
                    "recipe_metadata/format": self.format,
                }
            )
        )

        if self.card:
            self.standardization.steps.extend(self.card.preprocess_steps)

        self.processing.steps.append(self.task)

        if isinstance(self.augmentor, TaskInputsAugmentor):
            self.augmentor.set_fields(self.card.task.augmentable_inputs)
            self.processing.steps.append(self.augmentor)

        if self.has_custom_demos_pool:
            self.processing.steps.append(
                CreateDemosPool(
                    from_split=self.demos_taken_from,
                    to_split_names=[self.demos_pool_name, self.demos_taken_from],
                    to_split_sizes=[int(self.demos_pool_size)],
                    remove_targets_from_source_split=self.demos_removed_from_data,
                )
            )

        if self.use_demos:
            if self.sampler is None:
                if self.card.sampler is None:
                    raise ValueError(
                        "Unexpected None value for card.sampler. "
                        "To use num_demos > 0, please set a sampler on the TaskCard."
                    )
                self.sampler = self.card.sampler

        self.prepare_refiners()

        if self.use_demos:
            if isinstance(self.num_demos, int):
                self.verbalization.steps.append(
                    ConstantSizeSample(
                        from_stream=self.demos_pool_name,
                        to_field=self.demos_field,
                        sampler=self.sampler,
                        sample_size=self.num_demos,
                    )
                )
                self.verbalization.steps.append(
                    Set(fields={"recipe_metadata/num_demos": self.num_demos})
                )

            elif isinstance(self.num_demos, list):
                self.verbalization.steps.append(
                    RandomSizeSample(
                        from_stream=self.demos_pool_name,
                        to_field=self.demos_field,
                        sampler=self.sampler,
                        sample_sizes=self.num_demos,
                    )
                )
                self.verbalization.steps.append(
                    GetLength(field="demos", to_field="recipe_metadata/num_demos")
                )
            else:
                raise ValueError("num_demos must be int or List[int]")

            if isinstance(self.template, list):
                self.verbalization.steps.append(
                    ApplyRandomTemplate(
                        templates=self.template, demos_field=self.demos_field
                    )
                )
            else:
                self.verbalization.steps.append(
                    ApplySingleTemplate(
                        template=self.template, demos_field=self.demos_field
                    )
                )
        else:
            self.verbalization.steps.append(
                Set(fields={"recipe_metadata/num_demos": 0})
            )
            if isinstance(self.template, list):
                self.verbalization.steps.append(
                    ApplyRandomTemplate(templates=self.template)
                )
            else:
                self.verbalization.steps.append(
                    ApplySingleTemplate(template=self.template)
                )

        self.verbalization.steps.append(self.system_prompt)
        self.verbalization.steps.append(self.format)
        if isinstance(self.augmentor, FinalStateInputsAugmentor):
            self.verbalization.steps.append(self.augmentor)

        if self.postprocessors is not None:
            self.finalize.steps.append(
                Set(fields={"postprocessors": self.postprocessors})
            )

        if self.metrics is not None:
            self.finalize.steps.append(Set(fields={"metrics": self.metrics}))

        self.finalize.steps.append(Finalize(group_by=self.group_by))

    def prepare(self):
        if isinstance(self.template, TemplatesList):
            self.template = self.template.items
        self.reset_pipeline()


class StandardRecipeWithIndexes(BaseRecipe):
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
                    options = list(self.card.templates.keys())
                else:
                    options = list(range(0, len(self.card.templates)))
                raise ValueError(
                    f"card_template_index '{self.template_card_index}' is not defined in card. Possible card_template_index options: {options}"
                ) from e

        super().prepare()


class StandardRecipe(StandardRecipeWithIndexes):
    """This class represents a standard recipe for data processing and preparation.

    This class can be used to prepare a recipe.
    with all necessary steps, refiners and renderers included. It allows to set various
    parameters and steps in a sequential manner for preparing the recipe.

    Attributes:
        card (TaskCard): TaskCard object associated with the recipe.
        template (Template, optional): Template object to be used for the recipe.
        system_prompt (SystemPrompt, optional): SystemPrompt object to be used for the recipe.
        loader_limit (int, optional): Specifies the maximum number of instances per stream to be returned from the loader (used to reduce loading time in large datasets)
        format (SystemFormat, optional): SystemFormat object to be used for the recipe.
        metrics (List[str]): list of catalog metrics to use with this recipe.
        postprocessors (List[str]): list of catalog processors to apply at post processing. (Not recommended to use from here)
        group_by (List[Union[str, List[str]]]): list of task_data or metadata keys to group global scores by.
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
        demos_removed_from_data (bool, optional): whether to remove the demos from the source data, Default is True
        sampler (Sampler, optional): The Sampler used to select the demonstrations when num_demos > 0.
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
        AssertionError: If both template and template_card_index are specified at the same time.
    """

    pass
