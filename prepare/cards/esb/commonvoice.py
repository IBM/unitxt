from unitxt.audio_operators import ToAudio
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF, MultipleSourceLoader
from unitxt.operator import SourceSequentialOperator
from unitxt.operators import Rename
from unitxt.splitters import RenameSplits, SplitRandomMix
from unitxt.test_utils.card import test_card

# the datasets in 'esb/diagnostic-dataset' contain two splits: 'clean' and 'other'
# the script below downloads the two splits separately per dataset, then combines them into a single split named 'test'
card = TaskCard(
    loader=MultipleSourceLoader(
        sources=[
            SourceSequentialOperator(
                steps=[
                    LoadHF(
                        path="esb/diagnostic-dataset",
                        name="common_voice",
                        splits=["clean"],
                        data_classification_policy=["public"],
                        streaming=True,
                    ),
                    RenameSplits(
                        {
                            "clean": "test1",
                        }
                    ),
                ]
            ),
            SourceSequentialOperator(
                steps=[
                    LoadHF(
                        path="esb/diagnostic-dataset",
                        name="common_voice",
                        splits=["other"],
                        data_classification_policy=["public"],
                        streaming=True,
                    ),
                    RenameSplits(
                        {
                            "other": "test2",
                        }
                    ),
                ]
            ),
        ]
    ),
    preprocess_steps=[
        SplitRandomMix({"test": "test1+test2"}),
        ToAudio(field="audio"),
        Rename(field="norm_transcript", to_field="text"),
    ],
    task="tasks.speech_recognition",
    templates=[
        "templates.speech_recognition.default",
    ],
    __tags__={
        "license": "cc0-1.0",
        "language": "en",
        "task_categories": ["automatic-speech-recognition"],
        "size_categories": ["1K<n<10K"],
        "modalities": ["audio", "text"],
        "source": "European Parliament recordings",
        "benchmark": "ESB",
    },
    __description__=(
        "CommonVoice from ESB (End-to-End Speech Benchmark) - Common Voice is a series of crowd-sourced open-licensed speech datasets where speakers record text from Wikipedia in various languages."
    ),
    __title__="ESB-CommonVoice",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.esb.commonvoice", overwrite=True)
