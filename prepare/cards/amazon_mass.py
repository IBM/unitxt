from datasets import get_dataset_config_names

langs = get_dataset_config_names("AmazonScience/massive")
# now langs is the list of all languages showing in the dataset

from datasets import load_dataset_builder

ds_builder = load_dataset_builder("AmazonScience/massive")
classlabels = ds_builder.info.features["intent"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]
mappers = {"intent": mappers}

from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="AmazonScience/massive", name=lang),
        preprocess_steps=[
            MapInstanceValues(mappers=mappers),
            AddFields(
                fields={
                    "choices": classlabels.names,
                }
            ),
        ],
        task=FormTask(inputs=["choices", "utt"], outputs=["intent"], metrics=["metrics.accuracy"]),
        templates=TemplatesList(
            [
                InputOutputTemplate(
                    input_format="Given this sentence: {utt}. Classify it into one of the following classes. Choices: {choices}.",
                    output_format="{intent}",
                    postprocessors=["processors.take_first_non_empty_line"],
                )
            ]
        ),
    )
    if lang == "en-US":
        test_card(card, debug=True)
    filename = lang.replace("-", "_")
    add_to_catalog(card, f"cards.amazon_mass.{filename}", overwrite=True)
