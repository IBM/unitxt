import os

from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog

BLUEBENCH_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../../src/unitxt/catalog/recipes/bluebench",
)
bluebench_scenarios = {}

bluebench_directory = os.fsencode(BLUEBENCH_PATH)
for file in os.listdir(bluebench_directory):
    scenario_name = os.fsdecode(file)
    bluebench_scenarios[scenario_name] = {}
    for subscenario in os.listdir(f"{BLUEBENCH_PATH}/{scenario_name}"):
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
