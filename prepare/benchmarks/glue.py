from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "cola": DatasetRecipe(
            card="cards.cola",
            template="templates.classification.multi_class.instruction",
        ),
        "mnli": DatasetRecipe(
            card="cards.mnli",
            template="templates.classification.multi_class.relation.default",
        ),
        "mrpc": DatasetRecipe(
            card="cards.mrpc",
            template="templates.classification.multi_class.relation.default",
        ),
        "qnli": DatasetRecipe(
            card="cards.qnli",
            template="templates.classification.multi_class.relation.default",
        ),
        "rte": DatasetRecipe(
            card="cards.rte",
            template="templates.classification.multi_class.relation.default",
        ),
        "sst2": DatasetRecipe(
            card="cards.sst2", template="templates.classification.multi_class.title"
        ),
        "stsb": DatasetRecipe(
            card="cards.stsb", template="templates.regression.two_texts.title"
        ),
        "wnli": DatasetRecipe(
            card="cards.wnli",
            template="templates.classification.multi_class.relation.default",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.glue", overwrite=True)
