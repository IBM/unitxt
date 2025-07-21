from unitxt.audio_operators import ToAudio
from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

# LibriSpeech test.clean
card = TaskCard(
    loader=LoadHF(
        path="openslr/librispeech_asr",
        data_dir="clean",
        revision="refs/convert/parquet",
        splits=["test"],
        data_classification_policy=["public"],
        streaming=True,
    ),
    preprocess_steps=[
        ToAudio(field="audio"),
    ],
    task="tasks.speech_recognition",
    templates=[
        "templates.speech_recognition.default",
    ],
    __tags__={
        "license": "cc-by-4.0",
        "language": "en",
        "task_categories": ["automatic-speech-recognition"],
        "size_categories": ["1K<n<10K"],
        "modalities": ["audio", "text"],
        "source": "LibriVox audiobooks",
    },
    __description__=(
        "LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech "
        "derived from LibriVox audiobooks. This test.clean subset contains high-quality "
        "recordings from speakers with low Word Error Rate for clean speech recognition evaluation."
    ),
    __title__="LibriSpeech-test.clean",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.librispeech.test_clean", overwrite=True)

# LibriSpeech test
card = TaskCard(
    loader=LoadHF(
        path="openslr/librispeech_asr",
        data_dir="other",
        revision="refs/convert/parquet",
        splits=["test"],
        data_classification_policy=["public"],
        streaming=True,
    ),
    preprocess_steps=[
        ToAudio(field="audio"),
    ],
    task="tasks.speech_recognition",
    templates=[
        "templates.speech_recognition.default",
    ],
    __tags__={
        "license": "cc-by-4.0",
        "language": "en",
        "task_categories": ["automatic-speech-recognition"],
        "size_categories": ["1K<n<10K"],
        "modalities": ["audio", "text"],
        "source": "LibriVox audiobooks",
    },
    __description__=(
        "LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech "
        "derived from LibriVox audiobooks. This test subset contains recordings from all "
        "speakers including those with higher Word Error Rate for comprehensive evaluation."
    ),
    __title__="LibriSpeech-test",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.librispeech.test", overwrite=True)
