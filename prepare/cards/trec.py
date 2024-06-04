import sys

from datasets import load_dataset_builder
from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import Shuffle
from unitxt.test_utils.card import test_card

dataset_name = "trec"

ds_builder = load_dataset_builder(dataset_name)
classlabels = ds_builder.info.features["fine_label"]

expand_label_text = {
    "ABBR:abb": "Abbreviation: Abbreviation.",
    "ABBR:exp": "Abbreviation: Expression abbreviated.",
    "ENTY:animal": "Entity: Animal.",
    "ENTY:body": "Entity: Organ of body.",
    "ENTY:color": "Entity: Color.",
    "ENTY:cremat": "Entity: Invention, book and other creative piece.",
    "ENTY:currency": "Entity: Currency name.",
    "ENTY:dismed": "Entity: Disease and medicine.",
    "ENTY:event": "Entity: Event.",
    "ENTY:food": "Entity: Food.",
    "ENTY:instru": "Entity: Musical instrument.",
    "ENTY:lang": "Entity: Language.",
    "ENTY:letter": "Entity: Letter like a-z.",
    "ENTY:other": "Entity: Other entity.",
    "ENTY:plant": "Entity: Plant.",
    "ENTY:product": "Entity: Product.",
    "ENTY:religion": "Entity: Religion.",
    "ENTY:sport": "Entity: Sport.",
    "ENTY:substance": "Entity: Element and substance.",
    "ENTY:symbol": "Entity: Symbols and sign.",
    "ENTY:techmeth": "Entity: Techniques and method.",
    "ENTY:termeq": "Entity: Equivalent term.",
    "ENTY:veh": "Entity: Vehicle.",
    "ENTY:word": "Entity: Word with a special property.",
    "DESC:def": "Description: Definition of something.",
    "DESC:desc": "Description: Description of something.",
    "DESC:manner": "Description: Manner of an action.",
    "DESC:reason": "Description: Reason.",
    "HUM:gr": "Human: Group or organization of persons.",
    "HUM:ind": "Human: Individual.",
    "HUM:title": "Human: Title of a person.",
    "HUM:desc": "Human: Description of a person.",
    "LOC:city": "Location: City.",
    "LOC:country": "Location: Country.",
    "LOC:mount": "Location: Mountain.",
    "LOC:other": "Location: Other location.",
    "LOC:state": "Location: State.",
    "NUM:code": "Numeric: Postcode or other code.",
    "NUM:count": "Numeric: Number of something.",
    "NUM:date": "Numeric: Date.",
    "NUM:dist": "Numeric: Distance, linear measure.",
    "NUM:money": "Numeric: Price.",
    "NUM:ord": "Numeric: Order, rank.",
    "NUM:other": "Numeric: Other number.",
    "NUM:period": "Numeric: Lasting time of something",
    "NUM:perc": "Numeric: Percent, fraction.",
    "NUM:speed": "Numeric: Speed.",
    "NUM:temp": "Numeric: Temperature.",
    "NUM:volsize": "Numeric: Size, area and volume.",
    "NUM:weight": "Numeric: Weight.",
}

map_label_to_text = {
    str(i): expand_label_text[label] for i, label in enumerate(classlabels.names)
}
classes = [expand_label_text[label] for label in classlabels.names]

card = TaskCard(
    loader=LoadHF(path=dataset_name),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        SplitRandomMix(
            {"train": "train[85%]", "validation": "train[15%]", "test": "test"}
        ),
        RenameFields(field_to_field={"fine_label": "label"}),
        MapInstanceValues(mappers={"label": map_label_to_text}),
        AddFields(
            fields={
                "classes": classes,
                "text_type": "utterance",
                "type_of_class": "intent",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
        "annotations_creators": "expert-generated",
        "flags": ["croissant"],
        "language": "en",
        "language_creators": "expert-generated",
        "license": "unknown",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "1K<n<10K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": "multi-class-classification",
    },
    __description__=(
        "The Text REtrieval Conference (TREC) Question Classification dataset contains 5500 labeled questions in training set and another 500 for test set. The dataset has 6 coarse class labels and 50 fine class labels. Average length of each sentence is 10, vocabulary size of 8700. Data are collected from four sources: 4,500 English questions published by USC (Hovy et al., 2001), about 500 manually constructed questions for a few rare classes, 894 TREC 8 and TREC 9 questions, and also 500 questions from TREC 10 which serves as the test set. These questions were manually labeled."
    ),
)
test_card(card, debug=False)
add_to_catalog(artifact=card, name=f"cards.{dataset_name}", overwrite=True)
