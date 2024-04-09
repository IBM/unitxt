from datasets import get_dataset_config_names, load_dataset_builder
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

langs = get_dataset_config_names("AmazonScience/massive")
# now langs is the list of all languages showing in the dataset


ds_builder = load_dataset_builder("AmazonScience/massive")
classlabels = ds_builder.info.features["intent"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]


for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="AmazonScience/massive", name=lang),
        preprocess_steps=[
            MapInstanceValues(mappers={"intent": mappers}),
            RenameFields(field_to_field={"utt": "text", "intent": "label"}),
            AddFields(
                fields={
                    "classes": classlabels.names,
                    "text_type": "sentence",
                    "type_of_class": "intent",
                }
            ),
        ],
        task="tasks.classification.multi_class",
        templates="templates.classification.multi_class.all",
    )
    if lang == langs[0]:
        test_card(card, debug=False)
    filename = lang.replace("-", "_")
    add_to_catalog(card, f"cards.amazon_mass.{filename}", overwrite=True)
