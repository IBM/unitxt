from src.unitxt import load
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    SequentialRecipe,
    SplitRandomMix,
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
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "choices": ["entailment", "not entailment"],
            }
        ),
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
        SystemFormat(model_input_format="User: {source}\nAgent: "),
    ]
)

add_to_catalog(recipe, "recipes.wnli_fixed", overwrite=True)

dataset = load("recipes.wnli_fixed")

print_dict(dataset["train"][0])
