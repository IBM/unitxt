from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import StandardRecipe

benchmark = Benchmark(
    subsets={
        "cola": StandardRecipe(
            card="cards.cola",
            template="templates.classification.multi_class.instruction",
        ),
        "mnli": StandardRecipe(
            card="cards.mnli",
            template="templates.classification.multi_class.relation.default",
        ),
        "mrpc": StandardRecipe(
            card="cards.mrpc",
            template="templates.classification.multi_class.relation.default",
        ),
        "qnli": StandardRecipe(
            card="cards.qnli",
            template="templates.classification.multi_class.relation.default",
        ),
        "rte": StandardRecipe(
            card="cards.rte",
            template="templates.classification.multi_class.relation.default",
        ),
        "sst2": StandardRecipe(
            card="cards.sst2", template="templates.classification.multi_class.title"
        ),
        "stsb": StandardRecipe(
            card="cards.stsb", template="templates.regression.two_texts.title"
        ),
        "wnli": StandardRecipe(
            card="cards.wnli",
            template="templates.classification.multi_class.relation.default",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.glue", overwrite=True)
