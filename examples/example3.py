from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    RandomSampler,
    RenderTemplatedICL,
    SequentialRecipe,
    SliceSplit,
    SplitRandomMix,
    SpreadSplit,
    TextualInstruction,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.load import load_dataset
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
            }
        ),
        NormalizeListFields(fields=["choices"]),
        FormTask(
            inputs=["choices", "sentence1", "sentence2"],
            outputs=["label"],
            metrics=["metrics.accuracy"],
        ),
        SpreadSplit(
            source_stream="demos_pool",
            target_field="demos",
            sampler=RandomSampler(sample_size=5),
        ),
        RenderTemplatedICL(
            instruction=TextualInstruction(
                "classify if this sentence is entailment or not entailment."
            ),
            template=InputOutputTemplate(
                input_format="""
                    Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.
                """.strip(),
                output_format="{label}",
            ),
            demos_field="demos",
        ),
    ]
)

add_to_catalog(recipe, "recipes.wnli_5_shot", overwrite=True)

dataset = load_dataset("recipes.wnli_5_shot")

print_dict(dataset["train"][0])
