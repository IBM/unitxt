from src.unitxt import add_to_catalog
from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.operators import FilterByCondition
from src.unitxt.templates import InputOutputTemplate
from src.unitxt.test_utils.card import test_card

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
    templates=[
        InputOutputTemplate(  # based on "templates.classification.multi_class.default_no_instruction",
            input_format="Text: {text}",
            output_format="{label}",
            target_prefix="Topic: ",
            instruction="Classify the {type_of_class} of the following {text_type} to one of these options: {classes}.\n",
            postprocessors=[
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
        )
    ],
)

test_card(card, debug=False)
add_to_catalog(artifact=card, name=f"cards.{dataset_name}", overwrite=True)
