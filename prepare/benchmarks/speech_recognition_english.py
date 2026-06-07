from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "voxpopuli": DatasetRecipe(
            card="cards.esb.voxpopuli",
            format="formats.chat_api",
        ),
        "ami": DatasetRecipe(
            card="cards.esb.ami",
            format="formats.chat_api",
        ),
        "librispeech": DatasetRecipe(
            card="cards.esb.librispeech",
            format="formats.chat_api",
        ),
        "spgispeech": DatasetRecipe(
            card="cards.esb.spgispeech",
            format="formats.chat_api",
        ),
        "tedlium": DatasetRecipe(
            card="cards.esb.tedlium",
            format="formats.chat_api",
        ),
        "earnings22": DatasetRecipe(
            card="cards.esb.earnings22",
            format="formats.chat_api",
        ),
        "commonvoice": DatasetRecipe(
            card="cards.esb.commonvoice",
            format="formats.chat_api",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.speech_recognition", overwrite=True)
