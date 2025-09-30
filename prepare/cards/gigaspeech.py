from unitxt.audio_operators import ToAudio
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="speechcolab/gigaspeech",
        data_dir="xs",
        revision="refs/convert/parquet",
        splits=["train", "validation", "test"],
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
        "license": "apache-2.0",
        "language": "en",
        "task_categories": ["automatic-speech-recognition"],
        "size_categories": ["100K<n<1M"],
        "modalities": ["audio", "text"],
        "source": "audiobooks, podcasts, YouTube",
    },
    __description__=(
        "GigaSpeech is an evolving, multi-domain English speech recognition corpus with "
        "10,000 hours of high quality labeled audio suitable for supervised training. "
        "The transcribed audio data is collected from audiobooks, podcasts and YouTube, "
        "covering both read and spontaneous speaking styles, and a variety of topics "
        "such as arts, science, sports, etc."
    ),
    __title__="GigaSpeech-XS",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.gigaspeech.xs", overwrite=True)
