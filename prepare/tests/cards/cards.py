import json

from unitxt.artifact import fetch_artifact
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadFromDictionary

for card_name in [
    "wnli",
    "rte",
    "mmlu.marketing",
    "xwinogrande.pt",
    "sst2",
    "copa",
    "stsb",
    "almost_evil",
    "cola",
]:
    card, _ = fetch_artifact(f"cards.{card_name}")
    card.loader.loader_limit = 30
    data = card.loader().to_dataset()
    data = {k: v.to_list() for k, v in data.items()}

    card._init_dict["loader"] = json.loads(
        LoadFromDictionary(data=data, data_classification_policy=["public"]).to_json()
    )

    add_to_catalog(card, "cards.tests." + card_name, overwrite=True)
