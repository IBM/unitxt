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
for file in os.listdir(bluebench_dir):
    scenario_name = os.fsdecode(file)
    bluebench_scenarios[scenario_name] = {}
    for subscenario in os.listdir(os.path.join(bluebench_dir, scenario_name)):
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

benchmark = Benchmark(bluebench_scenarios)
add_to_catalog(benchmark, "benchmarks.bluebench", overwrite=True)
