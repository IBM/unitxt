from unitxt import load
from unitxt.artifact import Artifact
from unitxt.blocks import (
    AddFields,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    SequentialRecipe,
    SplitRandomMix,
    Task,
)
from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat
from unitxt.text_utils import print_dict

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
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "choices": ["entailment", "not entailment"],
                "instruction": "Classify the relationship between the two sentences from the choices.",
            }
        ),
        Task(
            inputs=["choices", "instruction", "sentence1", "sentence2"],
            outputs=["label"],
            metrics=["metrics.accuracy"],
        ),
        InputOutputTemplate(
            input_format="""
            Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.
            """.strip(),
            output_format="{label}",
        ),
        SystemFormat(model_input_format="User: {source}\nAgent: "),
    ]
)

assert isinstance(recipe, Artifact), "recipe must be an instance of Artifact"
add_to_catalog(recipe, "recipes.wnli", overwrite=True)
dataset = load("recipes.wnli")
print_dict(dataset["train"][0])
