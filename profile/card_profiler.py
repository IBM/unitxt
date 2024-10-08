import cProfile
import pstats
import sys
from io import StringIO

from unitxt.api import load_recipe
from unitxt.artifact import fetch_artifact
from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_settings
from unitxt.standard import StandardRecipe
from unitxt.stream import MultiStream
from unitxt.text_utils import print_dict

logger = get_logger()
settings = get_settings()
settings.allow_unverified_code = True


class CardProfiler:
    """Profiles the execution-time of api.load_dataset().

    Usage: set parameters: card, template, format, or any way you would invoke load_dateset() with,

    from unitxt root dir, run the following linux commands:

    (pip install snakeviz)
    python profile/card_profiler.py
    snakeviz profile/logs/benchmark_cards.prof

    An interactive browser window will open allowing to explore all time-details. See exporing options here:
    https://jiffyclub.github.io/snakeviz/
    (can also use the -s flag for snakeviz which will only set up a server and print out the url
    to use from another computer in order to view results shown by that server)

    look (ctrl-F) for methods named  profiler_...  to read profiling data for the major steps in the process
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

    def profiler_print_first_dicts(self, ms: MultiStream):
        logger.info(f"All {len(ms)} streams of generated ms:")
        for stream_name in ms:
            logger.info(f"{stream_name}")
        for stream_name in ms:
            instances = list(ms[stream_name])
            logger.info(
                f"First of all {len(instances)} instances of stream {stream_name}:"
            )
            print_dict(instances[0])

    def profiler_do_the_profiling(self, **kwargs):
        recipe = self.profiler_instantiate_recipe(**kwargs)
        ms = self.profiler_load_by_recipe(recipe)
        ms = self.profiler_metadata_and_standardization(ms, recipe)
        ms = self.profiler_processing_demos_metadata(ms, recipe)
        ms = self.profiler_verbalize_and_finalize(ms, recipe)
        self.profiler_print_first_dicts(ms)


def main_from_example():
    # copied from  examples/evaluate_a_judge_model_capabilities_on_arena_hard.py
    card = "cards.arena_hard.response_assessment.pairwise_comparative_rating.both_games_gpt_4_judge"
    template = "templates.response_assessment.pairwise_comparative_rating.arena_hard_with_shuffling"
    format = "formats.llama3_instruct"

    card_profiler = CardProfiler()
    card_profiler.profiler_do_the_profiling(card=card, template=template, format=format)


def main_from_cards():
    # cards = ["cards.cola", "cards.dart"]  # the benchmark
    cards = ["cards.cola"]  # the benchmark
    for card in cards:
        task_card, _ = fetch_artifact(card)
        template = task_card.templates.items[0]

        card_profiler = CardProfiler()
        card_profiler.profiler_do_the_profiling(card=task_card, template=template)


if __name__ == "__main__":
    cProfile.run(
        "main_from_cards()", "profile/logs/benchmark_cards.prof"
    )  # can change here to the other main_from_
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
    logger.info(
        f"tot={tot_time}, load={load_time}, diff={round(tot_time-load_time, 3)}"
    )
    sys.exit(round(tot_time - load_time, 3))
