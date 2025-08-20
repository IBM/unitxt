# This Python script is used to prepare cards for the CommonVoice ver. 17 dataset, used for evaluating multilingual speech recognition

from unitxt.audio_operators import ToAudio
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import Rename
from unitxt.string_operators import StripQuotation
from unitxt.test_utils.card import test_card

subsets = ["en", "fr", "de", "es", "pt"]  # languages to use
templates_ = {
    "en": "templates.speech_recognition.default",
    "fr": "templates.speech_recognition.multilingual",
    "de": "templates.speech_recognition.multilingual",
    "es": "templates.speech_recognition.multilingual",
    "pt": "templates.speech_recognition.multilingual",
}
tasks_ = {
    "en": "tasks.speech_recognition",
    "fr": "tasks.speech_recognition_multilingual",
    "de": "tasks.speech_recognition_multilingual",
    "es": "tasks.speech_recognition_multilingual",
    "pt": "tasks.speech_recognition_multilingual",
}

first = True
for subset in subsets:
    card = TaskCard(
        loader=LoadHF(
            path="mozilla-foundation/common_voice_17_0",
            revision="refs/convert/parquet",
            data_dir=subset,
            splits=["test"],
            data_classification_policy=["public"],
            streaming=True,
        ),
        preprocess_steps=[
            ToAudio(field="audio"),
            Rename(field="sentence", to_field="text"),
            StripQuotation(field="text"),
        ],
        task=tasks_[subset],
        templates=[templates_[subset]],
        __title__="CommonVoice-17-" + subset,
    )

    if first:
        test_card(card, strict=False)
        first = False
    add_to_catalog(card, f"cards.commonvoice.{subset}", overwrite=True)
