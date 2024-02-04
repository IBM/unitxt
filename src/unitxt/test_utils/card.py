import json
import math
import os.path
import tempfile

from .. import add_to_catalog, register_local_catalog
from ..artifact import fetch_artifact
from ..logging_utils import get_logger
from ..metric import _compute
from ..standard import StandardRecipe
from ..templates import TemplatesDict
from ..text_utils import construct_dict_str

logger = get_logger()
TEMP_NAME = "tmp_name"


def test_adding_to_catalog(card):
    with tempfile.TemporaryDirectory() as tmp_dir:
        add_to_catalog(
            card, TEMP_NAME, overwrite=True, catalog_path=tmp_dir, verbose=False
        )
        assert os.path.exists(
            os.path.join(tmp_dir, TEMP_NAME + ".json")
        ), "Card was not added to catalog"


def test_metrics_exist(card):
    for metric_name in card.task.metrics:
        metric, _ = fetch_artifact(metric_name)


def test_loading_from_catalog(card):
    with tempfile.TemporaryDirectory() as tmp_dir:
        add_to_catalog(
            card, TEMP_NAME, overwrite=True, catalog_path=tmp_dir, verbose=False
        )
        register_local_catalog(tmp_dir)
        card_, _ = fetch_artifact(TEMP_NAME)
        assert json.dumps(card_.to_dict(), sort_keys=True) == json.dumps(
            card.to_dict(), sort_keys=True
        ), "Card loaded is not equal to card stored"


def load_examples_from_standard_recipe(card, template_card_index, debug, **kwargs):
    disable = os.getenv("UNITXT_TEST_CARD_DISABLE", None)
    if disable is not None:
        logger.info(
            "load_examples_from_standard_recipe() functionality is disabled because UNITXT_TEST_CARD_DISABLE environment variable is set"
        )
        return None

    logger.info("*" * 80)
    logger.info(f"Using template card index: {template_card_index}")

    if "num_demos" not in kwargs:
        kwargs["num_demos"] = 3

    if "demos_pool_size" not in kwargs:
        kwargs["demos_pool_size"] = 20

    if "loader_limit" not in kwargs:
        kwargs["loader_limit"] = 200

    recipe = StandardRecipe(
        card=card, template_card_index=template_card_index, **kwargs
    )
    if debug:
        for max_steps in range(1, recipe.num_steps() + 1):
            examples = print_recipe_output(
                recipe,
                max_steps=max_steps,
                num_examples=1,
                print_header=True,
                print_stream_size=True,
            )
    else:
        examples = print_recipe_output(
            recipe,
            max_steps=recipe.num_steps(),
            num_examples=3,
            print_header=False,
            print_stream_size=False,
            streams=["test"],
        )
    return examples


def construct_recipe_output_message(
    recipe, max_steps, num_examples, print_header, print_stream_size, streams=None
):
    # Prepare the message string
    message = ""
    recipe.set_max_steps(max_steps)

    if print_header:
        step_description = recipe.get_last_step_description()
        header = "=" * 80 + "\n"
        header += "=" * 8 + "\n"
        header += "=" * 8 + " " + step_description + "\n"
        header += "=" * 8 + "\n"
        message += header

    multi_stream = recipe()

    if print_stream_size:
        for stream_name in multi_stream.keys():
            stream = multi_stream[stream_name]
            num_instances = len(list(iter(stream)))
            message += f"stream named '{stream_name}' has {num_instances} instances\n"
        message += "\n"

    examples = []
    for stream_name in multi_stream.keys():
        if streams is None or stream_name in streams:
            stream = multi_stream[stream_name]
            examples_in_stream = list(stream.take(num_examples))
            stream_header = "-" * 10 + "\n"
            stream_header += f"Showing {len(examples_in_stream)} example(s) from stream '{stream_name}':\n"
            message += stream_header

            for example in examples_in_stream:
                dict_message = construct_dict_str(example)
                message += dict_message + "\n\n"

            examples.extend(examples_in_stream)

    return message, examples


def print_recipe_output(
    recipe, max_steps, num_examples, print_header, print_stream_size, streams=None
):
    message, examples = construct_recipe_output_message(
        recipe, max_steps, num_examples, print_header, print_stream_size, streams
    )
    # Print the message
    message = "\n" + message
    logger.info(message)
    return examples


# flake8: noqa: C901
def test_with_eval(
    card,
    debug=False,
    strict=True,
    exact_match_score=1.0,
    full_mismatch_score=0.0,
    **kwargs,
):
    if type(card.templates) is TemplatesDict:
        for template_card_index in card.templates.keys():
            # restore the operators on the card, so that they are fresh for second invocation of
            # StandardRecipe as they are for the first one
            if card.preprocess_steps is not None:
                for step in card.preprocess_steps:
                    step.prepare()
            examples = load_examples_from_standard_recipe(
                card, template_card_index=template_card_index, debug=debug, **kwargs
            )
    else:
        num_templates = len(card.templates)
        for template_card_index in range(0, num_templates):
            # restore the operators on the card, so that they are fresh for second invocation of
            # StandardRecipe as they are for the first one
            if card.preprocess_steps is not None:
                for step in card.preprocess_steps:
                    step.prepare()
            examples = load_examples_from_standard_recipe(
                card, template_card_index=template_card_index, debug=debug, **kwargs
            )
    # metric = evaluate.load('unitxt/metric')
    correct_predictions = []
    for example in examples:
        correct_predictions.append(
            example["references"][0] if len(example["references"]) > 0 else []
        )

    results = _compute(predictions=correct_predictions, references=examples)
    score_name = results[0]["score"]["global"]["score_name"]
    score = results[0]["score"]["global"]["score"]

    if exact_match_score is not None and not math.isclose(score, exact_match_score):
        message = (
            f"The results of running the main metric used in the card ({score_name}) "
            f"over simulated predictions that are equal to the references returns a different score than expected.\n"
            f"One would expect a perfect score of {exact_match_score} in this case, but returned metric score was {score}.\n"
        )
        error_message = (
            f"{message}"
            f"This usually indicates an error in the metric or post processors, but can be also an acceptable edge case.\n"
            f"In anycase, this requires a review.  If this is acceptable, set strict=False in the call to test_card().\n"
            f"The predictions passed to the metrics were:\n {correct_predictions}\n"
        )
        warning_message = (
            f"{message}"
            f"This is flagged as only as a warning because strict=False was set in the call to test_card().\n"
            f"The predictions passed to the metrics were:\n {correct_predictions}\n"
        )
        if strict:
            raise AssertionError(error_message)
        logger.info("*" * 80)
        logger.info(warning_message)
        logger.info("*" * 80)

    wrong_predictions = ["a1s", "bfsdf", "dgdfgs", "gfjgfh", "ghfjgh"]
    results = _compute(predictions=wrong_predictions, references=examples)

    score_name = results[0]["score"]["global"]["score_name"]
    score = results[0]["score"]["global"]["score"]

    if full_mismatch_score is not None and score != full_mismatch_score:
        message = (
            f"The results of running the main metric used in the card ({score_name}) "
            f"over random predictions returns a different score than expected.\n"
            f"Usually, one would expect a low score of {full_mismatch_score} in this case, but returned metric score was {score}.\n"
        )
        error_message = (
            f"{message}"
            f"This can indicates an error in the metric or post processors, but can be also an acceptable edge case.\n"
            f"For example, in a metric that checks character level edit distance, a low none zero score is expected on random data.\n"
            f"In anycase, this requires a review.  If this is acceptable, set strict=False in the call to test_card().\n"
            f"The predictions passed to the metrics were:\n {wrong_predictions}\n"
        )
        warning_message = (
            f"{message}"
            f"This is flagged as only as a warning because strict=False was set in the call to test_card().\n"
            f"The predictions passed to the metrics were:\n {wrong_predictions}\n"
        )
        if strict:
            raise AssertionError(error_message)

        logger.info("*" * 80)
        logger.info(warning_message)
        logger.info("*" * 80)


def test_card(
    card,
    debug=False,
    strict=True,
    exact_match_score=1.0,
    full_mismatch_score=0.0,
    **kwargs,
):
    disable = os.getenv("UNITXT_TEST_CARD_DISABLE", None)
    if disable is not None:
        logger.info(
            "test_card() functionality is disabled because UNITXT_TEST_CARD_DISABLE environment variable is set"
        )
        return
    test_adding_to_catalog(card)
    test_metrics_exist(card)
    test_loading_from_catalog(card)
    test_with_eval(
        card,
        debug=debug,
        strict=strict,
        exact_match_score=exact_match_score,
        full_mismatch_score=full_mismatch_score,
        **kwargs,
    )
