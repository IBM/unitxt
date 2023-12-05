from src.unitxt.artifact import Artifact
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadHF,
    MapInstanceValues,
    RenderAutoFormatTemplate,
    SequentialRecipe,
    SplitRandomMix,
)
from src.unitxt.catalog import add_to_catalog

# from src.unitxt.catalog import add_to_catalog
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
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "choices": ["entailment", "not entailment"],
                "instruction": "classify the relationship between the two sentences from the choices.",
            }
        ),
        FormTask(
            inputs=["choices", "instruction", "sentence1", "sentence2"],
            outputs=["label"],
            metrics=["metrics.accuracy"],
        ),
        RenderAutoFormatTemplate(),
    ]
)

assert isinstance(recipe, Artifact), "Artifact must be an instance of Artifact"
add_to_catalog(recipe, "recipes.wnli", overwrite=True)
dataset = load_dataset("recipes.wnli")
print_dict(dataset["train"][0])
