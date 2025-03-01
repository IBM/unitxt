import argparse
import cProfile
import json
import os
import pstats
import tempfile
from collections import defaultdict
from io import StringIO
from time import time
from typing import Dict, Generator, Union

from unitxt.api import _source_to_dataset, evaluate, load_recipe
from unitxt.benchmark import Benchmark
from unitxt.dataclass import Dataclass
from unitxt.generator_utils import ReusableGenerator
from unitxt.inference import (
    CrossProviderInferenceEngine,
    InferenceEngine,
)
from unitxt.logging_utils import get_logger
from unitxt.operator import MultiStreamOperator
from unitxt.settings_utils import get_settings
from unitxt.standard import DatasetRecipe
from unitxt.stream import MultiStream, Stream

logger = get_logger()
settings = get_settings()

os.environ["UNITXT_MOCK_INFERENCE_MODE"] = "True"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"
os.environ["UNITXT_DISABLE_HF_DATASETS_CACHE"] = "False"

dataset_query = "benchmarks.bluebench[loader_limit=30,max_samples_per_subset=30,splits=[test]]"
# dataset_query = ["card=cards.cola", "card=cards.wnli"]
# dataset_query = "recipes.bluebench.knowledge.mmlu_pro_math"
# dataset_query = "card=cards.rag.documents.clap_nq.en"     # assaf's example
# dataset_query = "card=cards.cola"

class WaterMarkGenerator(Dataclass):
    measured_stream: Stream
    measured_stream_name: str
    water_mark: int = -1
    measuring_mode = True   # false: for blocking by the watermark

    def stream_counter(self)->Generator:
        gen = iter(self.measured_stream)
        if self.measuring_mode:
            for i, instance in enumerate (gen):
                if i > self.water_mark:
                    self.water_mark = i
                yield instance
        else:
            for i, instance in enumerate (gen):
                if i > self.water_mark:
                    break
                yield instance

class WaterMarkStreamer(MultiStreamOperator):

    measuring_mode: bool
    generators: Dict[str, WaterMarkGenerator] = None

    def process(self, multi_stream: MultiStream) -> MultiStream:
        if not self.generators:
            self.generators = {k: WaterMarkGenerator(measured_stream=multi_stream[k], measured_stream_name=k) for k in multi_stream}
        for gen_name in self.generators:
            self.generators[gen_name].measuring_mode = self.measuring_mode
        reusable_generators = {k: ReusableGenerator(generator=self.generators[k].stream_counter) for k in multi_stream}

        return MultiStream.from_generators(reusable_generators)


class BlueBenchProfiler:
    """Profiles the execution-time of loading, total-time (including loading) of recipe, inferenfe, and evaluate.

    goes by examples/evaluate_bluebench.py.

    Usage:

    from unitxt root dir, run the following linux commands:

    python performance/bluebench_profiler.py --output_file=<path_to_a_json_file> --employ_cPrile=<True or False>

    The script computes the total runtime of the dataset query hardcoded therein, and the time spent in loading the datasets,
    prepare it for inference (running throughout the recipes)
    then the inference of the overall dataset (made by grouping the many recipes products), and then
    the evaluation, and wraps all results into a json output_file, which is written in the path provided.

    Several example dataset queries are included in the script, of which exactly one is uncommented, which
    is the dataset evaluated. You can edit, change the dataset query per your need.

    If --output_file cmd line argument is not provided, the default path is taken to be 'performance/logs/bluebench.json'.

    In addition, if --employ_cProfile is True, the script generates a binary file named xxx.prof, as specified in field
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

    That detailed report is provided only for the
    """

    def equip_with_watermarker(self, recipe:Union[DatasetRecipe, Benchmark]):
        if isinstance(recipe, DatasetRecipe):
            water_mark_streamer = WaterMarkStreamer(measuring_mode=True)
            recipe.steps.insert(1, water_mark_streamer)
        else:
            # recipe is a benchmark
            for subset in recipe.subsets.values():
                self.equip_with_watermarker(subset)

    def change_watermarker_mode_to_block(self, recipe:Union[DatasetRecipe, Benchmark])->dict:
        if isinstance(recipe, DatasetRecipe):
            to_ret = {}
            if recipe.steps[1].generators:
                to_ret = {k: recipe.steps[1].generators[k].water_mark for k in recipe.steps[1].generators}
            recipe.steps[1].measuring_mode = False
            recipe.steps = recipe.steps[:2]
        else:
            # recipe is a benchmark
            to_ret = {}
            for subset_name in recipe.subsets:
                to_ret[subset_name] = self.change_watermarker_mode_to_block(recipe.subsets[subset_name])
        return to_ret

    def list_from_recipe_or_benchmark(self, recipe: Union[DatasetRecipe, Benchmark])-> dict:
        ms = recipe()
        return {k: len(list(ms[k])) for k in ms}

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

        # The official way -- based on api
        t0 = time()
        recipe = load_recipe(dataset_query, **kwargs)
        self.equip_with_watermarker(recipe)
        t1 = time()
        dataset = _source_to_dataset(source=recipe)
        t2 = time()
        dataset = self.list_from_dataset(dataset)
        t3 = time()
        model = self.profiler_instantiate_model()
        t4 = time()
        if isinstance(dataset, dict):
            if "test" in dataset:
                dataset = dataset["test"]
            else:
                split_name = next(iter(sorted(dataset.keys())))
                dataset = dataset[split_name]
        predictions = model.infer(dataset=dataset)
        t5 = time()
        evaluation_result = evaluate(predictions=predictions, data=dataset)
        t6 = time()
        # now just loading the data actually loaded above, and listing right after recipe.loader(), to report the loading time
        # from the total processing time.
        loaded_lengths = self.profiler_do_the_load_and_list_only(recipe=recipe)
        t7 = time()
        logger.critical(f"length of evaluation_result, over the returned dataset from Unitxt.load_dataset: {len(evaluation_result)}")
        logger.critical(f"lengths of (potentially fused) ingested datasets: {loaded_lengths}")

        return {
            "instantiate_recipe_from_query" : t1 - t0,
            "source_to_dataset": t2-t1,
            "list_out_dataset" : t3 - t2,
            "just_load_and_list": t7-t6,
            "instantiate_model": t4 - t3,
            "inference_time" : t5 - t4,
            "evaluation_time" : t6 - t5,
        }

    def profiler_do_the_load_and_list_only(self, recipe):
        water_marks = self.change_watermarker_mode_to_block(recipe)
        logger.critical(f"water marks = {water_marks}")
        return self.list_from_recipe_or_benchmark(recipe)


def profile_benchmark():
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
        "--employ_cProfiler",
        type=bool,
        default=False,
        help="whether to employ cProfile (True) or just time diffs(False). Defaults to False",
    )
    args = parser.parse_args()

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dict_to_print = {}

    if args.employ_cProfiler:
        # Create a temporary .prof file
        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as temp_prof_file:
            temp_prof_file_path = temp_prof_file.name
            cProfile.run("profile_benchmark()", temp_prof_file_path)

            f = StringIO()
            pst = pstats.Stats(temp_prof_file_path, stream=f)
            pst.strip_dirs()
            pst.sort_stats("name")  # sort by function name
            pst.print_stats(
                "profile_benchmark_blue_bench|profiler_unitxt_api_load_dataset_and_list|load_recipe|profiler_list_from_recipes_ms|profiler_instantiate_model|profiler_infer_predictions|profiler_evaluate_predictions|load_data|load_iterables|split_generator"
            )
            s = f.getvalue()
            assert s.split("\n")[7].split()[3] == "cumtime"
            instantiate_recipe_from_query = find_cummtime_of("load_recipe", "api.py", s)
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

        dict_to_print = {
            "instantiate_recipe_from_query" : instantiate_recipe_from_query,
            "source_to_dataset": source_to_dataset,
            "list_out_dataset" : list_out_dataset,
            "just_load_and_list": just_load_and_list,
            "instantiate_model": instantiate_model,
            "inference_time" : inference_time,
            "evaluation_time" : evaluation_time,
            "performance.prof file": temp_prof_file_path,

        }

    else:
        dict_to_print = profile_benchmark()

    dict_to_print["dataset_query"] = dataset_query
    dict_to_print["used_eager_mode"] = settings.use_eager_execution


    # Write the profiling results to the JSON file (user-specified)
    with open(args.output_file, "w+") as outfile:
        json.dump(dict_to_print, outfile)

    logger.info(f"JSON output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
