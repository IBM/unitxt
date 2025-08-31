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
                        name="spgispeech",
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
                        name="spgispeech",
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
        "license": "other",
        "language": "en",
        "task_categories": ["automatic-speech-recognition"],
        "size_categories": ["100K<n<1M"],
        "modalities": ["audio", "text"],
        "source": "financial earnings calls",
        "benchmark": "ESB",
    },
    __description__=(
        "SPGISpeech from ESB (End-to-End Speech Benchmark) - Financial earnings calls "
        "with standardized preprocessing and consistent formatting. Test-only split "
        "from the unified ESB dataset collection."
    ),
    __title__="ESB-SPGISpeech",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.esb.spgispeech", overwrite=True)
