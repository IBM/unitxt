from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "doc_vqa": DatasetRecipe(
            card="cards.doc_vqa.lmms_eval",
            template="templates.qa.llama_vision.with_context.doc_vqa",
            format="formats.chat_api",
        ),
        "info_vqa": DatasetRecipe(
            card="cards.info_vqa_lmms_eval",
            template="templates.qa.llama_vision.with_context.info_vqa",
            format="formats.chat_api",
        ),
        "chart_qa": DatasetRecipe(
            card="cards.chart_qa_lmms_eval",
            template="templates.qa.llama_vision.with_context.chart_qa",
            format="formats.chat_api",
        ),
        "ai2d": DatasetRecipe(
            card="cards.ai2d",
            template="templates.qa.llama_vision.multiple_choice.with_context.ai2d",
            format="formats.chat_api"
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.llama_vision", overwrite=True)
