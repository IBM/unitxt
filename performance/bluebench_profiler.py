import argparse
import cProfile
import json
import os
import pstats
import tempfile
from io import StringIO
from typing import Any, Dict, List, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from unitxt.api import evaluate, load_dataset, load_recipe
from unitxt.inference import (
    CrossProviderInferenceEngine,
    InferenceEngine,
    TextGenerationInferenceOutput,
)
from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_settings
from unitxt.stream import MultiStream

logger = get_logger()
settings = get_settings()

settings.allow_unverified_code = True
settings.disable_hf_datasets_cache = False
settings.mock_inference_mode = True


dataset_query = "benchmarks.bluebench[loader_limit=30,max_samples_per_subset=30,splits=[test]]"
# dataset_query = ["card=cards.cola", "card=cards.wnli"]
# dataset_query = "recipes.bluebench.knowledge.mmlu_pro_math"
# dataset_query = "card=cards.rag.documents.clap_nq.en"     # assaf's example

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

    def profiler_instantiate_recipe_result(
        self, dataset_query: str, **kwargs
    ) -> MultiStream:
        recipe = load_recipe(dataset_query, **kwargs)
        return recipe()

    def profiler_load_dataset(
        self, dataset_query: str, **kwargs
    ) -> Union[Dataset, IterableDataset, DatasetDict, IterableDatasetDict]:
        return load_dataset(dataset_query, **kwargs)

    def profiler_list_from_recipes_ms(self, ms)-> Dict[str, List[Dict[str, Any]]]:
        if not isinstance(ms, dict):
            to_return = list(ms)
            logger.critical(f"Listing {len(to_return)} instances from dataset.")
            return to_return

        to_return = {k: list(ms[k]) for k in ms}
        for k in to_return:
            logger.critical(f"Listing {len(to_return[k])} instances from Split '{k}'.")
        return to_return

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

    def profiler_do_the_profiling(self, dataset_query: str, **kwargs):
        logger.info(f"profiling the run of dataset_query = '{dataset_query}'")

        # first, the official way -- end-to-end
        dataset = self.profiler_load_dataset(
            dataset_query=dataset_query, **kwargs
        )

        if isinstance(dataset, dict):
            # continue with the first split can find:
            alternative_split = next(iter(sorted(dataset.keys())))
            dataset = dataset[alternative_split]

        if len(dataset) > 500:
            dataset = dataset[:500]

        model = self.profiler_instantiate_model()

        predictions = self.profiler_infer_predictions(model=model, dataset=dataset)

        evaluation_result = self.profiler_evaluate_predictions(
            predictions=predictions, dataset=dataset
        )
        logger.critical(f"length of evaluation_result, following Unitxt.load_dataset: {len(evaluation_result)}")

        # and now the old way, just to report time of generating a dataset, listed out from a ms
        ms = self.profiler_instantiate_recipe_result(
            dataset_query=dataset_query, **kwargs
        )

        dataset = self.profiler_list_from_recipes_ms(ms=ms)
        if not isinstance(dataset, dict):
            lengths = len(dataset)
        else:
            lengths = {k: len(dataset[k]) for k in dataset}

        logger.critical(f"length of recipe-result just listed: {lengths}")



def profile_benchmark_blue_bench():
    bluebench_profiler = BlueBenchProfiler()
    if isinstance(dataset_query, list):
        for dsq in dataset_query:
            bluebench_profiler.profiler_do_the_profiling(
            dataset_query=dsq
        )
    else:
        bluebench_profiler.profiler_do_the_profiling(
            dataset_query=dataset_query
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
            "profile_benchmark_blue_bench|profiler_load_dataset|profiler_instantiate_recipe_result|profiler_list_from_recipes_ms|profiler_instantiate_model|profiler_infer_predictions|profiler_evaluate_predictions|load_data|load_iterables|split_generator"
        )
        s = f.getvalue()
        assert s.split("\n")[7].split()[3] == "cumtime"
        load_dataset_time = find_cummtime_of(
            "profiler_load_dataset", "bluebench_profiler.py", s
        )
        # load_time = find_cummtime_of("load_data", "loaders.py", s)
        # load_time = find_cummtime_of(
        #     "load_iterables", "loaders.py", s
        # )
        # load_time += find_cummtime_of(
        #     "split_generator", "loaders.py", s
        # )
        instantiate_benchmark_time = find_cummtime_of(
            "profiler_instantiate_recipe_result", "bluebench_profiler.py", s
        )
        list_a_ms = find_cummtime_of(
            "profiler_list_from_recipes_ms", "bluebench_profiler.py", s
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
            "total_time": load_dataset_time,
            "instantiate_benchmark_time": instantiate_benchmark_time,
            "generate_benchmark_dataset_time": list_a_ms,
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
