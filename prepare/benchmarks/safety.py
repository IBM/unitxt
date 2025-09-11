from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "attaq": DatasetRecipe(card="cards.attaq"),
        "provoq": DatasetRecipe(card="cards.safety.provoq"),
        "airbench": DatasetRecipe(card="cards.safety.airbench2024"),
        "ailuminate": DatasetRecipe(card="cards.safety.mlcommons_ailuminate"),
    }
)

add_to_catalog(benchmark, "benchmarks.safety", overwrite=True)
