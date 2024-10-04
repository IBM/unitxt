from unitxt.api import load_recipe
from unitxt.artifact import fetch_artifact
from unitxt.logging_utils import get_logger
from unitxt.standard import StandardRecipe
from unitxt.stream import MultiStream
from unitxt.text_utils import print_dict

logger = get_logger()


class CardProfiler:
    """Profiles the execution-time of api.load_dataset().

    Usage: set parameters  card, template, format,   or any way you would invoke load_dateset() with,

    from unitxt root dir, run the following linux commands:

    (pip install snakeviz)
    python -m cProfile -o profile/logs/name_reflecting_parameters.prof profile/card_profiler.py
    snakeviz profile/logs/name_reflecting_parameters.prof

    A browser window will open with all time-details. See here how to explore these details:
    see https://jiffyclub.github.io/snakeviz/

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


def main_from_card():
    card = "cards.cola"
    task_card, _ = fetch_artifact(card)
    template = task_card.templates.items[0]

    card_profiler = CardProfiler()
    card_profiler.profiler_do_the_profiling(card=task_card, template=template)


if __name__ == "__main__":
    main_from_card()
