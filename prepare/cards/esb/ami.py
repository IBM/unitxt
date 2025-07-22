from unitxt.audio_operators import ToAudio
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="hf-audio/esb-datasets-test-only-sorted",
        name="ami",
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
        "size_categories": ["10K<n<100K"],
        "modalities": ["audio", "text"],
        "source": "AMI Meeting Corpus",
        "benchmark": "ESB",
    },
    __description__=(
        "AMI Meeting Corpus from ESB (End-to-End Speech Benchmark) - Meeting recordings "
        "with standardized preprocessing and consistent formatting. Test-only split from "
        "the unified ESB dataset collection."
    ),
    __title__="ESB-AMI",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.esb.ami", overwrite=True)
