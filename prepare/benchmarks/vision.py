from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "doc_vqa": DatasetRecipe(
            card="cards.doc_vqa.lmms_eval",
        ),
        "info_vqa": DatasetRecipe(
            card="cards.info_vqa_lmms_eval",
        ),
        "chart_qa": DatasetRecipe(
            card="cards.chart_qa_lmms_eval",
        ),
        "ai2d": DatasetRecipe(
            card="cards.ai2d",
        ),
        "websrc": DatasetRecipe(
            card="cards.websrc",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.vision", overwrite=True)
