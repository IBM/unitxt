import os
from collections import OrderedDict

from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.settings_utils import get_constants

constants = get_constants()

bluebench_dir = os.path.join(
    constants.catalog_dir,
    "recipes",
    "bluebench",
)

bluebench_scenarios = OrderedDict()

bluebench_scenarios = {}
for file in sorted(os.listdir(bluebench_dir)):
    scenario_name = os.fsdecode(file)
    bluebench_scenarios[scenario_name] = OrderedDict()
    for subscenario in sorted(os.listdir(os.path.join(bluebench_dir, scenario_name))):
        subscenario_name = os.fsdecode(subscenario)
        subscenario_name = (
            subscenario_name[: -len(".json")]
            if subscenario_name.endswith(".json")
            else subscenario_name
        )
        bluebench_scenarios[scenario_name][
            subscenario_name
        ] = f"recipes.bluebench.{scenario_name}.{subscenario_name}"
    bluebench_scenarios[scenario_name] = Benchmark(bluebench_scenarios[scenario_name])

benchmark = Benchmark(
    bluebench_scenarios,
    __description__=(
        "BlueBench is an open-source benchmark developed by domain experts to represent required needs of Enterprise users.\n\n"
        ".. image:: https://raw.githubusercontent.com/IBM/unitxt/main/assets/catalog/blue_bench_high_res_01.png\n"
        "   :alt: Optional alt text\n"
        "   :width: 30%\n"
        "   :align: center\n\n"
        "It is constructed using state-of-the-art benchmarking methodologies to ensure validity, robustness, and efficiency by utilizing unitxt's abilities for dynamic and flexible text processing.\n\n"
        "As a dynamic and evolving benchmark, BlueBench currently encompasses diverse domains such as legal, finance, customer support, and news. It also evaluates a range of capabilities, including RAG, pro-social behavior, summarization, and chatbot performance, with additional tasks and domains to be integrated over time."
    ),
)
add_to_catalog(benchmark, "benchmarks.bluebench", overwrite=True)
