from unitxt.audio_operators import ToAudio
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.test_utils.card import test_card

for subset in ["ihm", "sdm"]:
    card = TaskCard(
        loader=LoadHF(
            path="edinburghcstr/ami",
            data_dir=subset,
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
            "license": "cc-by-4.0",
            "language": "en",
            "task_categories": ["automatic-speech-recognition"],
            "size_categories": ["10K<n<100K"],
            "modalities": ["audio", "text"],
            "source": "AMI Meeting Corpus",
        },
        __description__=(
            "AMI Meeting Corpus contains 100 hours of meeting recordings with synchronized "
            "audio/video signals. The corpus includes close-talking and far-field microphones, "
            "recorded in 3 different rooms with varied acoustic properties. Features mostly "
            "non-native English speakers in meeting scenarios."
        ),
        __title__=f"AMI-{subset}",
    )

    test_card(card, strict=False)
    add_to_catalog(card, f"cards.ami.{subset}", overwrite=True)
