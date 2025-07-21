from unitxt.audio_operators import ToAudio
from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import Rename
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="kensho/spgispeech",
        data_dir="S",
        revision="refs/convert/parquet",
        splits=["train", "validation", "test"],
        data_classification_policy=["public"],
        streaming=True,
    ),
    preprocess_steps=[
        Rename(field="transcript", to_field="text"),
        ToAudio(field="audio"),
    ],
    task="tasks.speech_recognition",
    templates=[
        "templates.speech_recognition.default",
    ],
    __tags__={
        "license": "other",
        "language": "en",
        "task_categories": ["automatic-speech-recognition"],
        "size_categories": ["100K<n<1M"],
        "modalities": ["audio", "text"],
        "source": "financial earnings calls",
    },
    __description__=(
        "SPGISpeech is a large-scale transcription dataset of 5,000 hours of professionally-transcribed "
        "financial audio, containing earnings calls from 2007-2020. The dataset features business English "
        "with diverse accents from approximately 50,000 speakers, covering both spontaneous and narrated speech "
        "in 5-15 second segments."
    ),
    __title__="SPGISpeech-S",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.spgispeech.s", overwrite=True)
