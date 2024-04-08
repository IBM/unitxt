from unitxt import load
from unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    RandomSampler,
    SequentialRecipe,
    SliceSplit,
    SplitRandomMix,
    SpreadSplit,
)
from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat
from unitxt.logging_utils import get_logger

logger = get_logger()

recipe = SequentialRecipe(
    steps=[
        LoadHF(
            path="glue",
            name="wnli",
        ),
        SplitRandomMix(
            mix={
                "train": "train[95%]",
                "validation": "train[5%]",
                "test": "validation",
            }
        ),
        SliceSplit(
            slices={
                "demos_pool": "train[:100]",
                "train": "train[100:]",
                "validation": "validation",
                "test": "test",
            }
        ),
        MapInstanceValues(mappers={"label": {"0": "entails", "1": "not entails"}}),
        AddFields(
            fields={
                "choices": ["entails", "not entails"],
            }
        ),
        NormalizeListFields(fields=["choices"]),
        FormTask(
            inputs=["choices", "sentence1", "sentence2"],
            outputs=["label"],
            metrics=["metrics.accuracy"],
        ),
        InputOutputTemplate(
            input_format="""
                    Sentence1: {sentence1}\nSentence2: {sentence2}
                """.strip(),
            output_format="{label}",
            instruction="Classify the way Sentence1 relates to Sentence2, into the following two classes: [{choices}].\n",
            target_prefix="How relates: ",
        ),
        SpreadSplit(
            source_stream="demos_pool",
            target_field="demos",
            sampler=RandomSampler(sample_size=5),
        ),
        SystemFormat(
            demo_format="{source}\n{target_prefix}{target}\n\n",
            model_input_format="{instruction}\n{demos}\n{source}\n{target_prefix}",
        ),
    ]
)

add_to_catalog(recipe, "recipes.wnli_5_shot", overwrite=True)

dataset = load("recipes.wnli_5_shot")

logger.info(dataset.keys())
for i in range(5):
    logger.info(
        f"++++{i}++++{i}++++{i}++++{i}++++{i}++++{i}++++{i}++++{i}++++{i}++++{i}"
    )
    logger.info(dataset["train"][i]["source"])
