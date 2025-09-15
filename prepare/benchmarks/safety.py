from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "attaq": DatasetRecipe(
            card="cards.safety.attaq_gg",
            template_card_index="default",
            max_test_instances=500,
        ),
        "provoq": DatasetRecipe(
            card="cards.safety.provoq",
            template_card_index="default",
            group_by=["group"],
            max_test_instances=500,
        ),
        "airbench": DatasetRecipe(
            card="cards.safety.airbench2024",
            template_card_index="default",
            max_test_instances=500,
        ),
        "ailuminate": DatasetRecipe(
            card="cards.safety.mlcommons_ailuminate",
            template_card_index="default",
            max_test_instances=500,
        ),
    }
)

add_to_catalog(benchmark, "benchmarks.safety", overwrite=True)
