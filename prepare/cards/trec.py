import sys

from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    MapInstanceValues,
    Rename,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import Shuffle
from unitxt.test_utils.card import test_card

classes = [
    "ABBR:abb",
    "ABBR:exp",
    "ENTY:animal",
    "ENTY:body",
    "ENTY:color",
    "ENTY:cremat",
    "ENTY:currency",
    "ENTY:dismed",
    "ENTY:event",
    "ENTY:food",
    "ENTY:instru",
    "ENTY:lang",
    "ENTY:letter",
    "ENTY:other",
    "ENTY:plant",
    "ENTY:product",
    "ENTY:religion",
    "ENTY:sport",
    "ENTY:substance",
    "ENTY:symbol",
    "ENTY:techmeth",
    "ENTY:termeq",
    "ENTY:veh",
    "ENTY:word",
    "DESC:def",
    "DESC:desc",
    "DESC:manner",
    "DESC:reason",
    "HUM:gr",
    "HUM:ind",
    "HUM:title",
    "HUM:desc",
    "LOC:city",
    "LOC:country",
    "LOC:mount",
    "LOC:other",
    "LOC:state",
    "NUM:code",
    "NUM:count",
    "NUM:date",
    "NUM:dist",
    "NUM:money",
    "NUM:ord",
    "NUM:other",
    "NUM:period",
    "NUM:perc",
    "NUM:speed",
    "NUM:temp",
    "NUM:volsize",
    "NUM:weight",
]
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
    str(i): expand_label_text[label] for i, label in enumerate(classes)
}
classes = [expand_label_text[label] for label in classes]

card = TaskCard(
    loader=LoadHF(
        path="trec", revision="refs/convert/parquet", splits=["train", "test"]
    ),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        SplitRandomMix(
            {"train": "train[85%]", "validation": "train[15%]", "test": "test"}
        ),
        Rename(field_to_field={"fine_label": "label"}),
        MapInstanceValues(mappers={"label": map_label_to_text}),
        Set(
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
        "The Text REtrieval Conference (TREC) Question Classification dataset contains 5500 labeled questions in training set and another 500 for test set. \n"
        "The dataset has 6 coarse class labels and 50 fine class labels. Average length of each sentence is 10, vocabulary size of 8700… See the full description on the dataset page: https://huggingface.co/datasets/trec"
    ),
)
test_card(card, debug=False)
add_to_catalog(artifact=card, name="cards.trec", overwrite=True)
