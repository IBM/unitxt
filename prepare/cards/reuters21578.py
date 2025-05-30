from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    Rename,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.settings_utils import get_settings
from unitxt.test_utils.card import test_card

settings = get_settings()

dataset_name = "reuters21578"

classlabels = {
    "ModApte": [
        "acq",
        "alum",
        "austdlr",
        "barley",
        "bop",
        "can",
        "carcass",
        "castor-oil",
        "castorseed",
        "citruspulp",
        "cocoa",
        "coconut",
        "coconut-oil",
        "coffee",
        "copper",
        "copra-cake",
        "corn",
        "corn-oil",
        "cornglutenfeed",
        "cotton",
        "cotton-oil",
        "cottonseed",
        "cpi",
        "cpu",
        "crude",
        "cruzado",
        "dfl",
        "dkr",
        "dlr",
        "dmk",
        "earn",
        "f-cattle",
        "fishmeal",
        "fuel",
        "gas",
        "gnp",
        "gold",
        "grain",
        "groundnut",
        "groundnut-oil",
        "heat",
        "hog",
        "housing",
        "income",
        "instal-debt",
        "interest",
        "inventories",
        "ipi",
        "iron-steel",
        "jet",
        "jobs",
        "l-cattle",
        "lead",
        "lei",
        "lin-meal",
        "lin-oil",
        "linseed",
        "lit",
        "livestock",
        "lumber",
        "meal-feed",
        "money-fx",
        "money-supply",
        "naphtha",
        "nat-gas",
        "nickel",
        "nkr",
        "nzdlr",
        "oat",
        "oilseed",
        "orange",
        "palladium",
        "palm-oil",
        "palmkernel",
        "peseta",
        "pet-chem",
        "platinum",
        "plywood",
        "pork-belly",
        "potato",
        "propane",
        "rand",
        "rape-meal",
        "rape-oil",
        "rapeseed",
        "red-bean",
        "reserves",
        "retail",
        "rice",
        "ringgit",
        "rubber",
        "rupiah",
        "rye",
        "saudriyal",
        "sfr",
        "ship",
        "silver",
        "skr",
        "sorghum",
        "soy-meal",
        "soy-oil",
        "soybean",
        "stg",
        "strategic-metal",
        "sugar",
        "sun-meal",
        "sun-oil",
        "sunseed",
        "tapioca",
        "tea",
        "tin",
        "trade",
        "veg-oil",
        "wheat",
        "wool",
        "wpi",
        "yen",
        "zinc",
    ]
}
classlabels["ModLewis"] = classlabels["ModApte"]
classlabels["ModHayes"] = sorted(classlabels["ModApte"] + ["bfr", "hk"])

for subset in classlabels:
    card = TaskCard(
        loader=LoadHF(path=f"{dataset_name}", name=subset),
        preprocess_steps=[
            SplitRandomMix(
                {"train": "train[85%]", "validation": "train[15%]", "test": "test"}
            ),
            Rename(field_to_field={"topics": "labels"}),
            Set(fields={"classes": classlabels[subset], "type_of_classes": "topics"}),
        ],
        task="tasks.classification.multi_label",
        templates="templates.classification.multi_label.all",
        __tags__={"language": "en", "license": "other", "region": "us"},
        __description__=(
            "The Reuters-21578 dataset is one of the most widely used data collections for text categorization research. It is collected from the Reuters financial newswire service in 1987… See the full description on the dataset page: https://huggingface.co/datasets/reuters21578"
        ),
    )
    if subset == "ModHayes":
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.{dataset_name}.{subset}", overwrite=True)
