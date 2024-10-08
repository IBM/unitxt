import cProfile
import pstats
import shutil
from io import StringIO

import git
from unitxt.api import load_recipe
from unitxt.artifact import fetch_artifact
from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_settings
from unitxt.standard import StandardRecipe
from unitxt.stream import MultiStream
from unitxt.templates import TemplatesDict, TemplatesList
from unitxt.text_utils import print_dict

logger = get_logger()
settings = get_settings()
settings.allow_unverified_code = True


"""Profiles the execution-time of api.load_dataset(), over a benchmark of cards, comparing two branches.

Usage: set values for variables cards (the benchmark) and base_branch (typically, main) against which to compare runtime

from unitxt root dir, run the following linux commands:

(pip install GitPython)
python profile/card_profiler.py

The script computes and prints out the net runtime (total runtime minus loading time) of the benchmark cards in
the current branch, in the base_branch, and then divide the former by the latter and prints out the ratio of new to base.

Also, employing cPrifile, the script generates in profile/logs both performance profiles:
current_branch_benchmark_cards.prof  and  base_branch_benchmark_cards.prof
These profiles can be viewed and explored via:
(pip install snakeviz)
snakeviz profile/logs/current_branch_benchmark_cards.prof
and/or
snakeviz profile/logs/base_branch_benchmark_cards.prof

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
            f"All {len(ms)} streams of the multistream generated for card '{card_name}':"
        )
        for stream_name in ms:
            logger.info(f"{stream_name}")
        for stream_name in ms:
            instances = list(ms[stream_name])
            logger.info(
                f"The first of all {len(instances)} instances of stream {stream_name}:"
            )
            print_dict(instances[0])

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


def benchmark_net_time_per_branch() -> float:
    cProfile.run("profile_from_cards()", "profile/logs/benchmark_cards.prof")
    f = StringIO()
    pst = pstats.Stats("profile/logs/benchmark_cards.prof", stream=f)
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
    logger.info(f"tot={tot_time}, load={load_time}, diff={diff}")
    return diff


cards = ["cards.cola", "cards.dart"]  # the benchmark
base_branch_name = "main"

logger.info(f"benchmark cards are: {cards}")

repo = git.Repo(".")
current_branch = repo.active_branch
logger.info(
    f"Starting benchmark performance profiling in current branch, branch '{current_branch}'"
)
current_branch_net_runtime = benchmark_net_time_per_branch()
# copy the generated cprofile from profile/logs/benchmark_cards.prof to profile/logs/current_branch_benchmark_cards.prof
# so that it is not overwritten by the profiling info of the base_branch
shutil.copy(
    "profile/logs/benchmark_cards.prof",
    f"profile/logs/{current_branch}_benchmark_cards.prof",
)

base_branch = repo.heads[base_branch_name]
base_branch.checkout()
logger.info(
    f"Changed branch to '{repo.active_branch}', now start benchmark performance profiling in this branch"
)
base_branch_net_runtime = benchmark_net_time_per_branch()
# copy the generated cprofile from profile/logs/benchmark_cards.prof to profile/logs/current_branch_benchmark_cards.prof
# so that it is not overwritten by further branch profilings
shutil.copy(
    "profile/logs/benchmark_cards.prof",
    f"profile/logs/{base_branch}_benchmark_cards.prof",
)

logger.info(
    f"net run time (total minus loading) of benchmark in branch '{current_branch}' is {current_branch_net_runtime}"
)
logger.info(
    f"net run time (total minus loading) of benchmark in branch '{base_branch}' is {base_branch_net_runtime}"
)

ratio = round(current_branch_net_runtime / base_branch_net_runtime, 3)
logger.info(
    f"ratio of net runtimes of branches: '{current_branch}'/'{base_branch}' is {ratio}"
)

logger.info(f"Return to initial branch '{current_branch}'")
current_branch.checkout()
