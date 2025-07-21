from unitxt.audio_operators import ToAudio
from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import Rename
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="facebook/voxpopuli",
        data_dir="en",
        revision="refs/convert/parquet",
        splits=["train", "validation", "test"],
        data_classification_policy=["public"],
        streaming=True,
    ),
    preprocess_steps=[
        Rename(field="normalized_text", to_field="text"),
        ToAudio(field="audio"),
    ],
    task="tasks.speech_recognition",
    templates=[
        "templates.speech_recognition.default",
    ],
    __tags__={
        "license": "cc0-1.0",
        "language": "en",
        "task_categories": ["automatic-speech-recognition"],
        "multilinguality": "multilingual",
        "size_categories": ["1K<n<10K"],
        "modalities": ["audio", "text"],
        "source": "European Parliament recordings",
    },
    __description__=(
        "VoxPopuli is a large-scale multilingual speech corpus for representation learning, "
        "semi-supervised learning and interpretation. The raw data is collected from 2009-2020 "
        "European Parliament event recordings. This English subset contains transcribed speech data "
        "for automatic speech recognition with 1,791 hours of transcribed speech from 4,295 speakers."
    ),
    __title__="VoxPopuli-EN",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.voxpopuli.en", overwrite=True)
