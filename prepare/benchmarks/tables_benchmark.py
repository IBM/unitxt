import os
from collections import OrderedDict

from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.settings_utils import get_constants

constants = get_constants()

tables_benchmark_dir = os.path.join(
    constants.catalog_dir,
    "recipes",
    "tables_benchmark",
)


# Recursive function to build nested benchmarks
def build_nested_benchmark(dir_path, prefix="recipes.tables_benchmark"):
    nested_scenarios = OrderedDict()

    for entry in sorted(os.listdir(dir_path)):
        entry_path = os.path.join(dir_path, entry)
        entry_name = os.fsdecode(entry)

        if os.path.isdir(entry_path):  # Handle subdirectories
            # Recurse into subdirectory to create a nested benchmark
            sub_benchmark = build_nested_benchmark(entry_path, f"{prefix}.{entry_name}")
            nested_scenarios[entry_name] = sub_benchmark
        else:  # Handle individual JSON files
            scenario_name = (
                entry_name[: -len(".json")]
                if entry_name.endswith(".json")
                else entry_name
            )
            nested_scenarios[scenario_name] = f"{prefix}.{scenario_name}"

    # Create a Benchmark object for the current folder
    return Benchmark(nested_scenarios)


# Build the top-level benchmark
tables_benchmark_scenarios = build_nested_benchmark(tables_benchmark_dir)

benchmark = Benchmark(
    tables_benchmark_scenarios.subsets,
    __description__=(
        "TablesBenchmark is an open-source benchmark developed by domain experts to evaluate various table-related tasks and capabilities.\n\n"
        ".. image:: https://raw.githubusercontent.com/IBM/unitxt/main/assets/catalog/tables_benchmark.png\n"
        "   :alt: Optional alt text\n"
        "   :width: 30%\n"
        "   :align: center\n\n"
        "Constructed using state-of-the-art benchmarking methodologies, TablesBenchmark ensures validity, robustness, and efficiency by utilizing unitxt's dynamic and flexible text processing abilities.\n\n"
        "It encompasses diverse domains and evaluates a range of capabilities, with additional tasks and domains integrated over time."
    ),
)
add_to_catalog(benchmark, "benchmarks.tables_benchmark", overwrite=True)
