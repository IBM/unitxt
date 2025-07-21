from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "voxpopuli_en": DatasetRecipe(
            card="cards.voxpopuli.en",
            format="formats.chat_api",
        ),
        "ami_ihm": DatasetRecipe(
            card="cards.ami.ihm",
            format="formats.chat_api",
        ),
        "ami_sdm": DatasetRecipe(
            card="cards.ami.sdm",
            format="formats.chat_api",
        ),
        "gigaspeech_xs": DatasetRecipe(
            card="cards.gigaspeech.xs",
            format="formats.chat_api",
        ),
        "librispeech_test_clean": DatasetRecipe(
            card="cards.librispeech.test_clean",
            format="formats.chat_api",
        ),
        "librispeech_test": DatasetRecipe(
            card="cards.librispeech.test",
            format="formats.chat_api",
        ),
        "spgispeech_s": DatasetRecipe(
            card="cards.spgispeech.s",
            format="formats.chat_api",
        ),
        "tedlium_release1": DatasetRecipe(
            card="cards.tedlium.release1",
            format="formats.chat_api",
        ),
        "tedlium_release2": DatasetRecipe(
            card="cards.tedlium.release2",
            format="formats.chat_api",
        ),
        "tedlium_release3": DatasetRecipe(
            card="cards.tedlium.release3",
            format="formats.chat_api",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.speech_recognition", overwrite=True)
