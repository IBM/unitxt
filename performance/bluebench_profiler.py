import argparse
import cProfile
import json
import os
import pstats
import tempfile
from io import StringIO
from typing import Any, Dict, List, Union

from unitxt.api import evaluate, load_recipe
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    CrossProviderInferenceEngine,
    InferenceEngine,
    TextGenerationInferenceOutput,
)
from unitxt.logging_utils import get_logger
from unitxt.schema import UNITXT_DATASET_SCHEMA, loads_instance
from unitxt.settings_utils import get_settings

logger = get_logger()
settings = get_settings()
settings.allow_unverified_code = True


class BlueBenchProfiler:
    """Profiles the execution-time of loading, total-time (including loading) of recipe, inferenfe, and evaluate.

    goes by examples/evaluate_bluebench.py.

    Usage:

    from unitxt root dir, run the following linux commands:

    python performance/bluebench_profiler.py --output_file=<path_to_a_json_file>

    The script computes the total runtime of the benchmark, and the time spent in loading the datasets,
    prepare it for inference (running throughout the recipes)
    then the inference of the overall dataset (made by grouping the many recipes products), and then
    the evaluation, and wraps all results into a json output_file, which is written in the path provided.

    If --output_file cmd line argument is not provided, the default path is taken to be 'performance/logs/bluebench.json'.

    In addition, the script generates a binary file named xxx.prof, as specified in field
    "performance.prof file" of the json output_file,
    which can be nicely and interactively visualized via snakeviz:

    (pip install snakeviz)
    snakeviz <path provided in field 'performance.prof file' of the json output_file>

    snakeviz opens an interactive internet browser window allowing to explore all time-details.
    See exploring options here: https://jiffyclub.github.io/snakeviz/
    (can also use the -s flag for snakeviz which will only set up a server and print out the url
    to use from another computer in order to view results shown by that server)

    In the browser window, look (ctrl-F) for methods named  profiler_...  to read profiling data for the major steps in the process.
    You will find the total time of each step, accumulated over all recipes in the benchmark.
    """

    def profiler_instantiate_benchmark_recipe(
        self, dataset_query: str, **kwargs
    ) -> Benchmark:
        return load_recipe(dataset_query, **kwargs)

    def profiler_generate_benchmark_dataset(
        self, benchmark_recipe: Benchmark, split: str, **kwargs
    ) -> List[Dict[str, Any]]:
        with settings.context(
            disable_hf_datasets_cache=False,
            allow_unverified_code=True,
            mock_inference_mode=True,
        ):
            stream = benchmark_recipe()[split]

            dataset = stream.to_dataset(
                features=UNITXT_DATASET_SCHEMA, disable_cache=False
            ).with_transform(loads_instance)

            # to charge here for the time of generating all instances
            return list(dataset)

    def profiler_instantiate_model(self) -> InferenceEngine:
        return CrossProviderInferenceEngine(
            model="llama-3-8b-instruct",
            max_tokens=30,
        )

    def profiler_infer_predictions(
        self, model: InferenceEngine, dataset: List[Dict[str, Any]]
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return model.infer(dataset=dataset)

    def profiler_evaluate_predictions(self, predictions, dataset) -> dict:
        return evaluate(predictions=predictions, data=dataset)

    def profiler_do_the_profiling(self, dataset_query: str, split: str, **kwargs):
        benchmark_recipe = self.profiler_instantiate_benchmark_recipe(
            dataset_query=dataset_query, **kwargs
        )

        dataset = self.profiler_generate_benchmark_dataset(
            benchmark_recipe=benchmark_recipe, split=split, **kwargs
        )

        model = self.profiler_instantiate_model()

        predictions = self.profiler_infer_predictions(model=model, dataset=dataset)

        evaluation_result = self.profiler_evaluate_predictions(
            predictions=predictions, dataset=dataset
        )
        logger.critical(f"length of evaluation_result: {len(evaluation_result)}")


dataset_query = "benchmarks.bluebench[loader_limit=30,max_samples_per_subset=30]"


def profile_benchmark_blue_bench():
    bluebench_profiler = BlueBenchProfiler()
    bluebench_profiler.profiler_do_the_profiling(
        dataset_query=dataset_query, split="test"
    )


def find_cummtime_of(func_name: str, file_name: str, pst_printout: str) -> float:
    relevant_lines = list(
        filter(
            lambda x: f"({func_name})" in x and file_name in x,
            pst_printout.split("\n")[7:],
        )
    )
    if len(relevant_lines) == 0:
        return 0.0
    sumtimes = sum(
        round(float(relevant_line.split()[3]), 3) for relevant_line in relevant_lines
    )
    return round(sumtimes, 3)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Bluebench Profiler")
    parser.add_argument(
        "--output_file",
        type=str,
        default="performance/logs/bluebench.json",
        help="Path to save the json output file",
    )
    args = parser.parse_args()

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create a temporary .prof file
    with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as temp_prof_file:
        temp_prof_file_path = temp_prof_file.name
        cProfile.run("profile_benchmark_blue_bench()", temp_prof_file_path)

        f = StringIO()
        pst = pstats.Stats(temp_prof_file_path, stream=f)
        pst.strip_dirs()
        pst.sort_stats("name")  # sort by function name
        pst.print_stats(
            "profile_benchmark_blue_bench|profiler_instantiate_benchmark_recipe|profiler_generate_benchmark_dataset|profiler_instantiate_model|profiler_infer_predictions|profiler_evaluate_predictions|load_data|load_iterables"
        )
        s = f.getvalue()
        assert s.split("\n")[7].split()[3] == "cumtime"
        overall_tot_time = find_cummtime_of(
            "profile_benchmark_blue_bench", "bluebench_profiler.py", s
        )
        load_time = find_cummtime_of("load_data", "loaders.py", s)
        just_load_no_initial_ms_time = find_cummtime_of(
            "load_iterables", "loaders.py", s
        )
        instantiate_benchmark_time = find_cummtime_of(
            "profiler_instantiate_benchmark_recipe", "bluebench_profiler.py", s
        )
        generate_benchmark_dataset_time = find_cummtime_of(
            "profiler_generate_benchmark_dataset", "bluebench_profiler.py", s
        )
        instantiate_model_time = find_cummtime_of(
            "profiler_instantiate_model", "bluebench_profiler.py", s
        )
        inference_time = find_cummtime_of(
            "profiler_infer_predictions", "bluebench_profiler.py", s
        )
        evaluation_time = find_cummtime_of(
            "profiler_evaluate_predictions", "bluebench_profiler.py", s
        )

        # Data to be written
        dictionary = {
            "dataset_query": dataset_query,
            "total_time": overall_tot_time,
            "load_time": load_time,
            "load_time_no_initial_ms": just_load_no_initial_ms_time,
            "instantiate_benchmark_time": instantiate_benchmark_time,
            "generate_benchmark_dataset_time": generate_benchmark_dataset_time,
            "instantiate_model_time": instantiate_model_time,
            "inference_time": inference_time,
            "evaluation_time": evaluation_time,
            "used_eager_mode": settings.use_eager_execution,
            "performance.prof file": temp_prof_file_path,
        }

        # Write the profiling results to the JSON file (user-specified)
        with open(args.output_file, "w+") as outfile:
            json.dump(dictionary, outfile)

        logger.info(f"JSON output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
