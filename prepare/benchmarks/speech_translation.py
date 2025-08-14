from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

# running benchmarks with fleurs dataset, en-->xx
# running benchmarks with covost2 dataset, en-->xx
# running benchmarks with covost2 dataset, xx-->en
benchmark = Benchmark(
    subsets={
        "fleurs_en_de": DatasetRecipe(
            card="cards.fleurs.en_us.de_de",
            format="formats.chat_api",
        ),
        "fleurs_en_es": DatasetRecipe(
            card="cards.fleurs.en_us.es_419",
            format="formats.chat_api",
        ),
        "fleurs_en_fr": DatasetRecipe(
            card="cards.fleurs.en_us.fr_fr",
            format="formats.chat_api",
        ),
        "fleurs_en_it": DatasetRecipe(
            card="cards.fleurs.en_us.it_it",
            format="formats.chat_api",
        ),
        "fleurs_en_ja": DatasetRecipe(
            card="cards.fleurs.en_us.ja_jp",
            format="formats.chat_api",
        ),
        "fleurs_en_pt": DatasetRecipe(
            card="cards.fleurs.en_us.pt_br",
            format="formats.chat_api",
        ),
        "fleurs_en_zh": DatasetRecipe(
            card="cards.fleurs.en_us.cmn_hans_cn",
            format="formats.chat_api",
        ),
        "covost2_en_de": DatasetRecipe(
            card="cards.covost2.from_en.en_de",
            format="formats.chat_api",
        ),
        "covost2_en_ja": DatasetRecipe(
            card="cards.covost2.from_en.en_ja",
            format="formats.chat_api",
        ),
        "covost2_de_en": DatasetRecipe(
            card="cards.covost2.to_en.de_en",
            format="formats.chat_api",
        ),
        "covost2_es_en": DatasetRecipe(
            card="cards.covost2.to_en.es_en",
            format="formats.chat_api",
        ),
        "covost2_fr_en": DatasetRecipe(
            card="cards.covost2.to_en.fr_en",
            format="formats.chat_api",
        ),
        "covost2_pt_en": DatasetRecipe(
            card="cards.covost2.to_en.pt_en",
            format="formats.chat_api",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.speech_translation", overwrite=True)
