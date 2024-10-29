import argparse
import cProfile
import json
import os
import pstats
import tempfile
from io import StringIO

from unitxt.api import load_recipe
from unitxt.artifact import fetch_artifact
from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_settings
from unitxt.standard import StandardRecipe
from unitxt.stream import MultiStream
from unitxt.templates import TemplatesDict, TemplatesList

logger = get_logger()
settings = get_settings()
settings.allow_unverified_code = True


class CardProfiler:
    """Profiles the execution-time of api.load_dataset(), over a benchmark of cards.

    Usage: set values for variables cards (the benchmark)

    from unitxt root dir, run the following linux commands:

    python performance/card_profiler.py --output_file=<path_to_a_json_file>

    The script computes the total runtime of the benchmark, and the time spent in loading the dataset,
    accumulated across the cards in the benchmark, and wraps both results into a json output_file,
    which is written in the path provided.
    If --output_file cmd line argument is not provided, the default path is taken to be 'performance/logs/cards.json'.

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
    You will find the total time of each step, accumulated over all cards in the benchmark.
    """

    def profiler_instantiate_recipe(self, **kwargs) -> StandardRecipe:
        return load_recipe(**kwargs)

    def profiler_load_by_recipe(self, recipe: StandardRecipe) -> MultiStream:
        ms = recipe.loading.process()
        assert isinstance(ms, MultiStream)
        return ms

    def profiler_metadata_and_standardization(
        self, ms: MultiStream, recipe: StandardRecipe
    ) -> MultiStream:
        ms = recipe.metadata.process(ms)
        return recipe.standardization.process(ms)

    def profiler_processing_demos_metadata(
        self, ms: MultiStream, recipe: StandardRecipe
    ) -> MultiStream:
        ms = recipe.processing.process(ms)
        return recipe.metadata.process(ms)

    def profiler_verbalize_and_finalize(
        self, ms: MultiStream, recipe: StandardRecipe
    ) -> MultiStream:
        ms = recipe.verbalization.process(ms)
        return recipe.finalize.process(ms)

    def profiler_print_first_dicts(self, ms: MultiStream, card_name: str):
        logger.info(
            f"The multistream generated for card '{card_name}' has {len(ms)} streams of the following lengths:"
        )
        for stream_name in ms:
            logger.info(f"{stream_name} is of length {len(list(ms[stream_name]))}")

    def profiler_do_the_profiling(self, card_name: str, **kwargs):
        recipe = self.profiler_instantiate_recipe(**kwargs)
        ms = self.profiler_load_by_recipe(recipe)
        ms = self.profiler_metadata_and_standardization(ms, recipe)
        ms = self.profiler_processing_demos_metadata(ms, recipe)
        ms = self.profiler_verbalize_and_finalize(ms, recipe)
        self.profiler_print_first_dicts(ms, card_name)


def profile_from_cards():
    for card in cards:
        task_card, _ = fetch_artifact(card)
        if isinstance(task_card.templates, TemplatesList):
            template = task_card.templates.items[0]
        elif isinstance(task_card.templates, list):
            template = task_card.templates[0]
        elif isinstance(task_card, TemplatesDict):
            for templ in task_card.templates.items.values():
                template = templ
                break
        else:
            raise ValueError(
                f"Unidentified type of templates {task_card.templates} in card {card}"
            )

        card_profiler = CardProfiler()
        card_profiler.profiler_do_the_profiling(
            card_name=card, card=task_card, template=template, loader_limit=5000
        )


cards = ["cards.cola", "cards.dart"]  # the benchmark


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Card Profiler")
    parser.add_argument(
        "--output_file",
        type=str,
        default="performance/logs/cards.json",
        help="Path to save the json output file",
    )
    args = parser.parse_args()

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"benchmark cards are: {cards}")

    # Create a temporary .prof file
    with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as temp_prof_file:
        temp_prof_file_path = temp_prof_file.name
        cProfile.run("profile_from_cards()", temp_prof_file_path)

        f = StringIO()
        pst = pstats.Stats(temp_prof_file_path, stream=f)
        pst.strip_dirs()
        pst.sort_stats("name")  # sort by function name
        pst.print_stats("profiler_do_the_profiling|profiler_load_by_recipe")
        s = f.getvalue()
        assert s.split("\n")[7].split()[3] == "cumtime"
        assert "profiler_do_the_profiling" in s.split("\n")[8]
        tot_time = round(float(s.split("\n")[8].split()[3]), 3)
        assert "profiler_load_by_recipe" in s.split("\n")[9]
        load_time = round(float(s.split("\n")[9].split()[3]), 3)
        diff = round(tot_time - load_time, 3)

        # Data to be written
        dictionary = {
            "total_time": tot_time,
            "load_time": load_time,
            "net_time": diff,
            "cards_tested": cards,
            "used_eager_mode": settings.use_eager_execution,
            "performance.prof file": temp_prof_file_path,
        }

        # Write the profiling results to the JSON file (user-specified)
        with open(args.output_file, "w+") as outfile:
            json.dump(dictionary, outfile)

        logger.info(f"JSON output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
