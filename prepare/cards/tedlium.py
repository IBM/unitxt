from unitxt.audio_operators import ToAudio
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.test_utils.card import test_card

for release in ["release1", "release2", "release3"]:
    card = TaskCard(
        loader=LoadHF(
            path="LIUM/tedlium",
            data_dir=release,
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
            "license": "cc-by-nc-nd-3.0",
            "language": "en",
            "task_categories": ["automatic-speech-recognition"],
            "size_categories": ["10K<n<100K"],
            "modalities": ["audio", "text"],
            "source": "TED Talks",
        },
        __description__=(
            "The TED-LIUM corpus is English-language TED talks, with transcriptions, sampled at 16kHz. "
            f"This {release} contains progressively more transcribed speech training data, ranging from "
            "118 hours (Release 1), to 207 hours (Release 2), to 452 hours (Release 3). The dataset "
            "includes speaker information and is commonly used for automatic speech recognition evaluation."
        ),
        __title__=f"TEDLIUM-{release}",
    )

    test_card(card, strict=False)
    add_to_catalog(card, f"cards.tedlium.{release}", overwrite=True)
