# This Python script is used to prepare card for the covost2 dataset, used for evaluating speech translation

import os

from unitxt.audio_operators import ToAudio
from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import Set
from unitxt.test_utils.card import test_card

# The covost2 dataset supports translation from 21 languages into English, and from English to 15 languages
# As opposed to the fluers dataset, there is no support from any to any

# Entire list of supported language pairs (codes) for reference:
# all_subsets_from_en = [
#    "de",
#    "tr",
#    "fa",
#    "sv-SE",
#    "mn",
#    "zh-CN",
#    "cy",
#    "ca",
#    "sl",
#    "et",
#    "id",
#    "ar",
#    "ta",
#    "lv",
#    "ja"
#    ]

# all_subsets_to_en = [
#    "fr",
#    "de",
#    "es",
#    "ca",
#    "it",
#    "ru",
#    "zh-CN",
#    "pt",
#    "fa",
#    "et",
#    "mn",
#    "nl",
#    "tr",
#    "ar",
#    "sv-SE",
#    "lv",
#    "sl",
#    "ta",
#    "ja",
#    "id",
#    "cy"
#    ]

# Currently we only use covost2 for evaluating translating to and from english for a limited list as below; additions may follow
# We use the (basic) Whisper text normalization; before extending to other languages, check if Whisper basic normalizer supports them

subsets_from_en = ["de", "ja"]

subsets_to_en = ["fr", "de", "es", "pt"]

lang_name = {
    "de": "German",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
    "en": "English",
}

# An example of how to load the covost2 dataset using Hugging Face's loader:
#   dataset = datasets.load_dataset('facebook/covost2', 'en_de', data_dir='/dccstor/aharonsatt/data/covost2/en', split='test')
#
# For each language pair, the source dataset (the one that provides the audio) needs to be available locally
# We are temporarily using the data in the CCC, at the path /dataset/speechdata/CoVoST2/xx  where xx is the source language code

task_types = ["from_en", "to_en"]
local_data_path = "/dataset/speechdata/CoVoST2/"  # the CommonVoice ver. 4 datasets are stored locally in the CCC; this local store is required for using HF CoVost 2 dataset

first = True

# from English
for lang in subsets_from_en:
    card = TaskCard(
        loader=LoadHF(
            path="facebook/covost2",
            name="en_" + lang,
            data_dir=local_data_path + "en",
            split="test",
            streaming=True,
            requirements=["datasets<4.0.0"],
        ),
        preprocess_steps=[
            ToAudio(field="audio"),
            Set(
                fields={
                    "source_language": lang_name["en"],
                    "target_language": lang_name[lang],
                }
            ),
        ],
        task="tasks.translation.speech",
        templates=[
            "templates.translation.speech.default",
        ],
    )

    if first:
        test_card(card, debug=True)
        first = False

    add_to_catalog(card, f"cards.covost2.from_en.en_{lang}", overwrite=True)

# to English
for lang in subsets_to_en:
    card = TaskCard(
        loader=LoadHF(
            path="facebook/covost2",
            name=lang + "_en",
            data_dir=local_data_path + lang,
            split="test",
            streaming=True,
        ),
        preprocess_steps=[
            ToAudio(field="audio"),
            Set(
                fields={
                    "source_language": lang_name[lang],
                    "target_language": lang_name["en"],
                }
            ),
        ],
        task="tasks.translation.speech",
        templates=[
            "templates.translation.speech.default",
        ],
    )

    if first and os.path.isdir(local_data_path):
        test_card(card, demos_taken_from="test", num_demos=0)
        first = False

    add_to_catalog(card, f"cards.covost2.to_en.{lang}_en", overwrite=True)
