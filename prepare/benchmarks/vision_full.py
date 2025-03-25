from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

benchmark = Benchmark(
    subsets={
        "doc_vqa_default": DatasetRecipe(
            card="cards.doc_vqa.lmms_eval",
        ),
        "info_vqa_default": DatasetRecipe(
            card="cards.info_vqa_lmms_eval",
        ),
        "chart_qa_default": DatasetRecipe(
            card="cards.chart_qa_lmms_eval",
        ),
        "ai2d_default": DatasetRecipe(
            card="cards.ai2d",
        ),
        "websrc_default": DatasetRecipe(
            card="cards.websrc",
        ),
        "doc_vqa_llama_vision_template": DatasetRecipe(
            card="cards.doc_vqa.lmms_eval",
            template="templates.qa.llama_vision.with_context.doc_vqa",
            format="formats.chat_api",
        ),
        "info_vqa_llama_vision_template": DatasetRecipe(
            card="cards.info_vqa_lmms_eval",
            template="templates.qa.llama_vision.with_context.info_vqa",
            format="formats.chat_api",
        ),
        "chart_qa_llama_vision_template": DatasetRecipe(
            card="cards.chart_qa_lmms_eval",
            template="templates.qa.llama_vision.with_context.chart_qa",
            format="formats.chat_api",
        ),
        "ai2d_llama_vision_template": DatasetRecipe(
            card="cards.ai2d",
            template="templates.qa.llama_vision.multiple_choice.with_context.ai2d",
            format="formats.chat_api",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.vision_full", overwrite=True)
