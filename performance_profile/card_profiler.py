import cProfile
import json
import pstats
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

"""Profiles the execution-time of api.load_dataset(), over a benchmark of cards.

Usage: set values for variables cards (the benchmark)

from unitxt root dir, run the following linux commands:

python performance_profile/card_profiler.py

The script computes the total runtime of the benchmark, and the time spent in loading the dataset,
accumulated across the cards in the benchmark, and wraps both results into a json file:
performance_profile/logs/cards_benchmark.json

In addition, the script generates a binary file named performance_profile/logs/cards_benchmark.prof,
which can be nicely and interactively visualized via snakeviz:

(pip install snakeviz)
snakeviz performance_profile/logs/cards_benchmark.prof

snakeviz opens an interactive internet browser window allowing to explore all time-details.
See exporing options here: https://jiffyclub.github.io/snakeviz/
(can also use the -s flag for snakeviz which will only set up a server and print out the url
to use from another computer in order to view results shown by that server)

In the browser window, look (ctrl-F) for methods named  profiler_...  to read profiling data for the major steps in the process.
You will find the total time of each step, accumulated along all cards in the benchmark.
"""


class CardProfiler:
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
    logger.info(f"benchmark cards are: {cards}")

    cProfile.run(
        "profile_from_cards()", "performance_profile/logs/cards_benchmark.prof"
    )
    f = StringIO()
    pst = pstats.Stats("performance_profile/logs/cards_benchmark.prof", stream=f)
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
    }
    with open("performance_profile/logs/cards_benchmark.json", "w") as outfile:
        json.dump(dictionary, outfile)


if __name__ == "__main__":
    main()
