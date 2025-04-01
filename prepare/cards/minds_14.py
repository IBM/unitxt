from unitxt.audio_operators import ToAudio
from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import MapInstanceValues, Rename
from unitxt.splitters import SplitRandomMix
from unitxt.test_utils.card import test_card

classes = [
    "abroad",
    "address",
    "app_error",
    "atm_limit",
    "balance",
    "business_loan",
    "card_issues",
    "cash_deposit",
    "direct_debit",
    "freeze",
    "high_value_payment",
    "joint_account",
    "latest_transactions",
    "pay_bill",
]

card = TaskCard(
    loader=LoadHF(path="PolyAI/minds14", name="en-US"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}
        ),
        MapInstanceValues(mappers={"intent_class": {str(i): label for i, label in enumerate(classes)}}),
        Rename(field="intent_class", to_field="label"),
        Set(
            fields={
                "text_type": "sentence",
                "type_of_class": "intent",
                "classes": classes
            }
        ),
        ToAudio(field="audio", to_field="text"),

    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
    "annotations_creators": [
        "expert-generated",
        "crowdsourced",
        "machine-generated"
    ],
    "language_creators": [
        "crowdsourced",
        "expert-generated"
    ],
    "language": [
        "en", "fr", "it", "es", "pt", "de", "nl", "ru", "pl", "cs", "ko", "zh"
    ],
    "license": "cc-by-4.0",
    "multilinguality": "multilingual",
    "size_categories": "10K<n<100K",
    "task_categories": "automatic-speech-recognition",
    "task_ids": "keyword-spotting",
    "pretty_name": "MInDS-14",
    "language_bcp47": [
        "en", "en-GB", "en-US", "en-AU", "fr", "it", "es", "pt", "de", "nl", "ru", "pl", "cs", "ko", "zh"
    ],
    "tags": ["speech-recognition"]
    },
    __description__=(
        "MINDS-14 is training and evaluation resource for intent detection task with spoken data. It covers 14 intents extracted from a commercial system in the e-banking domain, associated with spoken examples in 14 diverse language varieties."
    ),
)

test_card(card)
add_to_catalog(card, "cards.minds_14", overwrite=True)
