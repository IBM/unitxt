from src.unitxt import load
from src.unitxt.blocks import (
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
    TextualInstruction,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat
from src.unitxt.text_utils import print_dict

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
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "choices": ["entailment", "not entailment"],
                "instance": TextualInstruction(
                    "classify if this sentence is entailment or not entailment."
                ),
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
                    Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.
                """.strip(),
            output_format="{label}",
        ),
        SpreadSplit(
            source_stream="demos_pool",
            target_field="demos",
            sampler=RandomSampler(sample_size=5),
        ),
        SystemFormat(
            demo_format="User: {source}\nAgent: {target}\n\n",
            model_input_format="{demos}User: {source}\nAgent: ",
        ),
    ]
)

add_to_catalog(recipe, "recipes.wnli_5_shot", overwrite=True)

dataset = load("recipes.wnli_5_shot")

print_dict(dataset["train"][0])
