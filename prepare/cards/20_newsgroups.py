from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import FilterByCondition
from unitxt.test_utils.card import test_card

dataset_name = "20_newsgroups"

map_labels = {
    "alt.atheism": "atheism",
    "comp.graphics": "computer graphics",
    "comp.os.ms-windows.misc": "microsoft windows",
    "comp.sys.ibm.pc.hardware": "pc hardware",
    "comp.sys.mac.hardware": "mac hardware",
    "comp.windows.x": "windows x",
    "misc.forsale": "for sale",
    "rec.autos": "cars",
    "rec.motorcycles": "motorcycles",
    "rec.sport.baseball": "baseball",
    "rec.sport.hockey": "hockey",
    "sci.crypt": "cryptography",
    "sci.electronics": "electronics",
    "sci.med": "medicine",
    "sci.space": "space",
    "soc.religion.christian": "christianity",
    "talk.politics.guns": "guns",
    "talk.politics.mideast": "middle east",
    "talk.politics.misc": "politics",
    "talk.religion.misc": "religion",
}

card = TaskCard(
    loader=LoadHF(path=f"SetFit/{dataset_name}", streaming=True),
    preprocess_steps=[
        FilterByCondition(values={"text": ""}, condition="ne"),
        SplitRandomMix(
            {"train": "train[90%]", "validation": "train[10%]", "test": "test"}
        ),
        RenameFields(field_to_field={"label_text": "label"}),
        MapInstanceValues(mappers={"label": map_labels}),
        AddFields(
            fields={
                "classes": list(map_labels.values()),
                "text_type": "text",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={"flags": ["croissant"], "region": "us"},
    __description__=(
        "This is a version of the 20 newsgroups dataset that is provided in Scikit-learn. From the Scikit-learn docs:\n"
        "The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). The split between the train and test set is based upon a messages posted before and after a specific date.\n"
        "We followed the recommended practice to remove headers, signature blocks, and… See the full description on the dataset page: https://huggingface.co/datasets/SetFit/20_newsgroups."
    ),
)

test_card(card, debug=False)
add_to_catalog(artifact=card, name=f"cards.{dataset_name}", overwrite=True)
