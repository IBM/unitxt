from unitxt.audio_operators import ToAudio
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="hf-audio/esb-datasets-test-only-sorted",
        name="gigaspeech",
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
        "license": "apache-2.0",
        "language": "en",
        "task_categories": ["automatic-speech-recognition"],
        "size_categories": ["100K<n<1M"],
        "modalities": ["audio", "text"],
        "source": "audiobooks, podcasts, YouTube",
        "benchmark": "ESB",
    },
    __description__=(
        "GigaSpeech from ESB (End-to-End Speech Benchmark) - Multi-domain English speech "
        "corpus with standardized preprocessing and consistent formatting. Test-only split "
        "from the unified ESB dataset collection."
    ),
    __title__="ESB-GigaSpeech",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.esb.gigaspeech", overwrite=True)
