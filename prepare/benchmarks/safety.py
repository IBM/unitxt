from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

MAX_TEST_INSTANCES = 1000

benchmark = Benchmark(
    subsets={
        "attaq": DatasetRecipe(
            card="cards.safety.attaq_gg",
            template_card_index="default",
            group_by=["label"],
            max_test_instances=MAX_TEST_INSTANCES,
        ),
        "provoq": DatasetRecipe(
            card="cards.safety.provoq_gg",
            template_card_index="default",
            group_by=["group"],
            max_test_instances=MAX_TEST_INSTANCES,
        ),
        "airbench": DatasetRecipe(
            card="cards.safety.airbench2024",
            template_card_index="default",
            group_by=["l2-name"],
            max_test_instances=MAX_TEST_INSTANCES,
        ),
        "ailuminate": DatasetRecipe(
            card="cards.safety.mlcommons_ailuminate",
            template_card_index="default",
            group_by=["hazard"],
            max_test_instances=MAX_TEST_INSTANCES,
        ),
    }
)

add_to_catalog(benchmark, "benchmarks.safety", overwrite=True)
