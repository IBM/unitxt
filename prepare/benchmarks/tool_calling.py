from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe

subsets = {}

for bfcl_subset in [
    "simple",
    "multiple",
    "live_multiple",
    "live_simple",
    "java",
    "javascript",
    "parallel",
    "parallel_multiple",
    "live_parallel",
    "live_parallel_multiple",
]:
    subsets[f"bfcl.{bfcl_subset}"] = DatasetRecipe(
        card=f"cards.bfcl.multi_turn.{bfcl_subset}_v3",
        format="formats.chat_api",
        metrics=[
            "metrics.tool_calling.multi_turn.validity",
            "metrics.tool_calling.multi_turn.correctness.llama_3_3_70b_instruct_judge",
        ],
    )

subsets["xlam"] = DatasetRecipe(
    card="cards.xlam_function_calling_60k",
    format="formats.chat_api",
    metrics=[
        "metrics.tool_calling.multi_turn.validity",
        "metrics.tool_calling.multi_turn.correctness.llama_3_3_70b_instruct_judge",
    ],
)


benchmark = Benchmark(subsets=subsets)

add_to_catalog(benchmark, "benchmarks.tool_calling", overwrite=True)
