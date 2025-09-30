from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "commonvoice_de": DatasetRecipe(
            card="cards.commonvoice.de",
            format="formats.chat_api",
        ),
        "commonvoice_es": DatasetRecipe(
            card="cards.commonvoice.es",
            format="formats.chat_api",
        ),
        "commonvoice_fr": DatasetRecipe(
            card="cards.commonvoice.fr",
            format="formats.chat_api",
        ),
        "commonvoice_pt": DatasetRecipe(
            card="cards.commonvoice.pt",
            format="formats.chat_api",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.speech_recognition_multilingual", overwrite=True)
