from unitxt.audio_operators import ToAudio
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="hf-audio/esb-datasets-test-only-sorted",
        name="librispeech",
        splits=["test.clean"],
        data_classification_policy=["public"],
        streaming=True,
    ),
    preprocess_steps=[
        RenameSplits({"test.clean": "test"}),
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
        "benchmark": "ESB",
    },
    __description__=(
        "LibriSpeech from ESB (End-to-End Speech Benchmark) - Read English speech from "
        "audiobooks with standardized preprocessing and consistent formatting. Test-only "
        "split from the unified ESB dataset collection."
    ),
    __title__="ESB-LibriSpeech",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.esb.librispeech", overwrite=True)
