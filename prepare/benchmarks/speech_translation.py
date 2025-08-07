from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "en_de": DatasetRecipe(
            card="cards.fleurs.en_us.de_de",
            format="formats.chat_api",
        ),
        "en_es": DatasetRecipe(
            card="cards.fleurs.en_us.es_419",
            format="formats.chat_api",
        ),
        "en_fr": DatasetRecipe(
            card="cards.fleurs.en_us.fr_fr",
            format="formats.chat_api",
        ),
        "en_it": DatasetRecipe(
            card="cards.fleurs.en_us.it_it",
            format="formats.chat_api",
        ),
        "en_ja": DatasetRecipe(
            card="cards.fleurs.en_us.ja_jp",
            format="formats.chat_api",
        ),
        "en_pt": DatasetRecipe(
            card="cards.fleurs.en_us.pt_br",
            format="formats.chat_api",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.speech_translation", overwrite=True)
