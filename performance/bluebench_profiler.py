import argparse
import cProfile
import json
import os
import pstats
import tempfile
from collections import defaultdict
from io import StringIO
from time import time
from typing import Dict, Generator

from unitxt.api import _source_to_dataset, evaluate, load_dataset, load_recipe
from unitxt.dataclass import Dataclass
from unitxt.generator_utils import ReusableGenerator
from unitxt.inference import (
    CrossProviderInferenceEngine,
    InferenceEngine,
)
from unitxt.logging_utils import get_logger
from unitxt.operator import MultiStreamOperator
from unitxt.settings_utils import get_settings
from unitxt.stream import MultiStream, Stream

logger = get_logger()
settings = get_settings()

os.environ["UNITXT_MOCK_INFERENCE_MODE"] = "True"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"
os.environ["UNITXT_DISABLE_HF_DATASETS_CACHE"] = "False"

dataset_query = "benchmarks.bluebench[loader_limit=30,splits=[test]]"
# dataset_query = ["card=cards.cola", "card=cards.wnli"]
# dataset_query = "recipes.bluebench.knowledge.mmlu_pro_math"
# dataset_query = "card=cards.rag.documents.clap_nq.en"     # assaf's example
# dataset_query = "card=cards.cola"

class WaterMarkGenerator(Dataclass):
    measured_stream: Stream
    water_mark: int = -1

    def stream_counter(self)->Generator:
        for i, instance in enumerate (self.measured_stream):
            if i > self.water_mark:
                self.water_mark = i
            yield instance

class WaterMarkStreamer(MultiStreamOperator):

    generators: Dict[str, WaterMarkGenerator] = None

    def process(self, multi_stream: MultiStream) -> MultiStream:
        if not self.generators:
            self.generators = {k: WaterMarkGenerator(measured_stream=multi_stream[k]) for k in multi_stream}
        reusable_generators = {k: ReusableGenerator(generator=self.generators[k].stream_counter) for k in multi_stream}

        return MultiStream.from_generators(reusable_generators)


class BlueBenchProfiler:
    """Profiles the execution-time of loading, total-time (including loading) of recipe, inferenfe, and evaluate.

    goes by examples/evaluate_bluebench.py.

    Usage:

    from unitxt root dir, run the following linux commands:

    python performance/bluebench_profiler.py --output_file=<path_to_a_json_file> --employ_cProfile

    The script computes the total runtime of the dataset query hardcoded therein, specifying the time spent in loading the datasets,
    prepare it for inference (running throughout the recipes)
    then the inference of the overall dataset (in case of benchmark -- that overall dataset is made by grouping the many
    recipes products), and then the evaluation, and wraps all results into a json output_file, which is written in the path provided.

    Several example dataset queries are included in the script, of which exactly one is to be uncommented, which
    is the dataset evaluated. You can edit, change the dataset query per your need.
    A list of dataset_queries is also allowed, showing as one of the examples. The reported times in this case
    are accumulated over all the members of the list.

    If --output_file cmd line argument is not provided, the default path is taken to be 'performance/logs/bluebench.json'.

    In addition, if --employ_cProfile is used, the script generates a binary file named xxx.prof, as specified in field
    "performance.prof file" of the json output_file,
    which can be nicely and interactively visualized via snakeviz:

    (pip install snakeviz)
    snakeviz <path provided in field 'performance.prof file' of the json output_file>

    snakeviz opens an interactive internet browser window allowing to explore all time-details.
    See exploring options here: https://jiffyclub.github.io/snakeviz/
    (can also use the -s flag for snakeviz which will only set up a server and print out the url
    to use from another computer in order to view results shown by that server)

    In the browser window, you can look (ctrl-F) for any unitxt function name, or the profile_ functions in this script, to read profiling data for the major steps in the process.
    You can also sort by time. You will find the total time of each step (function), accumulated over all recipes in the benchmark.

    """

    def list_from_dataset(self, dataset):
        if not isinstance(dataset, dict):
            return list(dataset)
        return {k: list(dataset[k]) for k in dataset}

    def profiler_instantiate_model(self) -> InferenceEngine:
        return CrossProviderInferenceEngine(
            model="llama-3-8b-instruct",
            max_tokens=30,
        )

    def profiler_do_the_profiling(self, dataset_query: str, **kwargs):
        logger.info(f"profiling the run of dataset_query = '{dataset_query}'")

        # The official way -- based on api.py
        current_loader_cache_size = settings.loader_cache_size
        settings.loader_cache_size = 1000   # cover all needs
        t0 = time()
        recipe = load_recipe(dataset_query, **kwargs)
        t0_5 = time()
        #exhaust recipe to ensure all data is loaded, and hopefully saved in the expanded loader-cache
        ms = recipe()
        for stream_name in ms:
            len(list(ms[stream_name]))
        dataset = _source_to_dataset(source=recipe)
        # and now begin again, with same recipe in which the loaders enjoy the cache
        t1 = time()
        ms = recipe()
        t2 = time()
        dataset = _source_to_dataset(source=recipe)
        t3 = time()
        model = self.profiler_instantiate_model()
        t4 = time()
        # reduce to one split, and infer
        if isinstance(dataset, dict):
            if "test" in dataset:
                dataset = dataset["test"]
            else:
                split_name = next(iter(sorted(dataset.keys())))
                dataset = dataset[split_name]
        predictions = model.infer(dataset=dataset)
        t5 = time()
        evaluate(predictions=predictions, data=dataset)
        t6 = time()
        # now just streaming through recipe, without generating an HF dataset:
        ms = recipe()
        total_production_length_of_recipe = {k: len(list(ms[k])) for k in ms}
        logger.critical(f"lengths of total production of recipe: {total_production_length_of_recipe}")

        settings.loader_cache_size = current_loader_cache_size
        return {
            "load_recipe" : t0_5 - t0,
            "recipe()": t2 - t1,
            "source_to_dataset": t3-t2,
            "instantiate_model": t4 - t3,
            "inference_time" : t5 - t4,
            "evaluation_time" : t6 - t5,
            "load_data_time": t1-t0
        }

def profile_benchmark():
    # profile the official way, with cProfile
    bluebench_profiler = BlueBenchProfiler()
    queries = dataset_query if isinstance(dataset_query, list) else [dataset_query]
    for dsq in queries:
        dataset = load_dataset(dataset_query=dsq)
        model = bluebench_profiler.profiler_instantiate_model()
        if isinstance(dataset, dict):
            if "test" in dataset:
                dataset = dataset["test"]
            else:
                split_name = next(iter(sorted(dataset.keys())))
                dataset = dataset[split_name]
        predictions = model.infer(dataset=dataset)
        evaluation_result = evaluate(predictions=predictions, data=dataset)
        logger.info(f"length of evaluation result from cProfiling: {len(evaluation_result)}")

def profile_no_cprofile():
    bluebench_profiler = BlueBenchProfiler()
    queries = dataset_query if isinstance(dataset_query, list) else [dataset_query]
    res = defaultdict(float)
    for dsq in queries:
        dsq_time = bluebench_profiler.profiler_do_the_profiling(
            dataset_query=dsq
        )
        for k in dsq_time:
            res[k] += dsq_time[k]
    return {k: round(res[k], 3) for k in res}

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


def use_cprofile():
    # Create a temporary .prof file
    with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as temp_prof_file:
        temp_prof_file_path = temp_prof_file.name
        cProfile.run("profile_benchmark()", temp_prof_file_path)

        f = StringIO()
        pst = pstats.Stats(temp_prof_file_path, stream=f)
        pst.strip_dirs()
        pst.sort_stats("name")  # sort by function name
        pst.print_stats(
            "profile_benchmark|load_dataset|load_recipe|profiler_do_the_load_and_list_only|profiler_instantiate_model|infer|evaluate|load_data|load_iterables|split_generator"
        )
        s = f.getvalue()
        assert s.split("\n")[7].split()[3] == "cumtime"
        load_recipe = find_cummtime_of("load_recipe", "api.py", s)
        source_to_dataset = find_cummtime_of("_source_to_dataset", "api.py", s)
        list_out_dataset = find_cummtime_of("list_from_dataset", "bluebench_profiler.py", s)
        just_load_and_list = find_cummtime_of(
            "profiler_do_the_load_and_list_only", "bluebench_profiler.py", s
        )
        instantiate_model = find_cummtime_of(
            "profiler_instantiate_model", "bluebench_profiler.py", s
        )
        inference_time = find_cummtime_of(
            "infer", "inference.py", s
        )
        evaluation_time = find_cummtime_of(
            "evaluate", "api.py", s
        )

    return {
        "load_recipe" : load_recipe,
        "source_to_dataset": source_to_dataset,
        "list_out_dataset" : list_out_dataset,
        "just_load_and_list": just_load_and_list,
        "instantiate_model": instantiate_model,
        "inference_time" : inference_time,
        "evaluation_time" : evaluation_time,
        "performance.prof file": temp_prof_file_path,
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Bluebench Profiler")
    parser.add_argument(
        "--output_file",
        type=str,
        default="performance/logs/bluebench.json",
        help="Path to save the json output file",
    )
    parser.add_argument(
        "--employ_cProfile",
        action="store_true",
        help="whether to employ cProfile or just time diffs.",
    )
    args = parser.parse_args()

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dict_to_print = profile_no_cprofile()

    cprofile_dict = None
    if args.employ_cProfile:
        cprofile_dict = use_cprofile()

    if cprofile_dict:
        dict_to_print["performance.prof file"] = cprofile_dict["performance.prof file"]

    dict_to_print["dataset_query"] = dataset_query
    dict_to_print["used_eager_mode"] = settings.use_eager_execution


    # Write the profiling results to the JSON file (user-specified)
    with open(args.output_file, "w+") as outfile:
        json.dump(dict_to_print, outfile)

    logger.info(f"JSON output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
