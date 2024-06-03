import json
import math
import os.path
import tempfile

from .. import add_to_catalog, register_local_catalog
from ..artifact import fetch_artifact
from ..logging_utils import get_logger
from ..metric import _compute
from ..settings_utils import get_settings
from ..standard import StandardRecipe
from ..templates import TemplatesDict
from ..text_utils import construct_dict_str

logger = get_logger()
settings = get_settings()

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
    if settings.test_card_disable:
        logger.info(
            "load_examples_from_standard_recipe() functionality is disabled because unitxt.settings.test_card_disable=True or UNITXT_TEST_CARD_DISABLE environment variable is set"
        )
        return None

    logger.info("*" * 80)
    if "loader_limit" not in kwargs:
        kwargs["loader_limit"] = 30
    kwargs["template_card_index"] = template_card_index

    recipe = StandardRecipe(card=card, **kwargs)
    logger.info(f"Using these card recipe parameters: {kwargs}")

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
            num_examples=5,
            print_header=False,
            print_stream_size=False,
            streams=["test"],
        )
    return examples


def print_recipe_output(
    recipe, max_steps, num_examples, print_header, print_stream_size, streams=None
):
    recipe.set_max_steps(max_steps)

    if print_header:
        step_description = recipe.get_last_step_description()
        logger.info("=" * 80)
        logger.info("=" * 8)
        logger.info(f"{'=' * 8} {step_description}")
        logger.info("=" * 8)

    multi_stream = recipe()

    if print_stream_size:
        for stream_name in multi_stream.keys():
            stream = multi_stream[stream_name]
            num_instances = len(list(iter(stream)))
            logger.info(f"stream named '{stream_name}' has {num_instances} instances\n")

    examples = []
    for stream_name in multi_stream.keys():
        if streams is None or stream_name in streams:
            stream = multi_stream[stream_name]
            logger.info("-" * 10)
            logger.info(
                f"Showing up to {num_examples} examples from stream '{stream_name}':"
            )
            for example, _ in zip(stream, range(num_examples)):
                dict_message = construct_dict_str(example)
                logger.info(dict_message)
                logger.info("\n")
                examples.append(example)
    return examples


def print_predictions(correct_predictions, results):
    for result, correct_prediction in zip(results, correct_predictions):
        logger.info("*" * 5)
        logger.info(
            f"Prediction: ({type(correct_prediction).__name__})     {correct_prediction}"
        )
        logger.info(
            f"Processed prediction: ({type(result['prediction']).__name__}) {result['prediction']}"
        )
        logger.info(
            f"Processed references: ({type(result['references']).__name__}) {result['references']}"
        )
    logger.info("*" * 5)
    logger.info("Score output:")
    logger.info(json.dumps(results[0]["score"], sort_keys=True, indent=4))


def test_correct_predictions(examples, strict, exact_match_score):
    correct_predictions = [example["target"] for example in examples]
    logger.info("*" * 40)
    logger.info("Running with the gold references as predictions.")
    results = _compute(predictions=correct_predictions, references=examples)

    logger.info("Showing the output of the post processing:")

    print_predictions(correct_predictions, results)

    score_name = results[0]["score"]["global"]["score_name"]
    score = results[0]["score"]["global"]["score"]

    if not math.isclose(score, exact_match_score):
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
        logger.info("*" * 10)
        logger.info(warning_message)


def test_wrong_predictions(
    examples, strict, maximum_full_mismatch_score, full_mismatch_prediction_values
):
    import random

    wrong_predictions = [
        random.choice(full_mismatch_prediction_values) for example in examples
    ]
    logger.info("*" * 40)
    logger.info("Running with random values as predictions.")
    results = _compute(predictions=wrong_predictions, references=examples)

    logger.info("Showing the output of the post processing:")
    print_predictions(wrong_predictions, results)

    score_name = results[0]["score"]["global"]["score_name"]
    score = results[0]["score"]["global"]["score"]

    if score > maximum_full_mismatch_score:
        message = (
            f"The results of running the main metric used in the card ({score_name}) "
            f"over random predictions returns a different score than expected.\n"
            f"The test expected a low score of atmost {maximum_full_mismatch_score} in this case, but returned metric score was {score}.\n"
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

        logger.info("*" * 10)
        logger.info(warning_message)


def test_card(
    card,
    debug=False,
    strict=True,
    test_exact_match_score_when_predictions_equal_references=True,
    test_full_mismatch_score_with_full_mismatch_prediction_values=True,
    exact_match_score=1.0,
    maximum_full_mismatch_score=0.0,
    full_mismatch_prediction_values=None,
    **kwargs,
):
    """Tests a given card.

    By default, the test goes over all templates defined in the card,
    and generates sample outputs for template. It also runs two tests on sample data.
    The first is running the metrics in the card with predictions which are equal to the references.
    The expected score in this case is typically 1.  The second test is running the metrics in the card
    with random predictions (selected from a fixed set of values).  The score expected in this case
    is typically 0.

    During the test, sample datasets instances, as well as the predictions/references are displayed.
    It also shows the processed predictions and references, after the template's post processors
    are applied.  Thus wayit is possible to debug and see that the inputs to the metrics are as expected.

        Parameters:
        1. `card`: The `Card` object to be tested.
        2. `debug`: A boolean value indicating whether to enable debug mode. In debug mode, the data processing pipeline is executed step by step, printing a representative output of each step.  Default is False.
        3. `strict`: A boolean value indicating whether to fail if scores do not match the expected ones.
           Default is True.
        4. `test_exact_match_score_when_predictions_equal_references`: A boolean value indicating whether to test the exact match score when predictions equal references. Default is True.
        5. `test_full_mismatch_score_with_full_mismatch_prediction_values`: A boolean value indicating whether to test the full mismatch score with full mismatch prediction values.
            The potential mismatched predeiction values are specified in full_mismatch_prediction_values`.
            Default is True.
        6. `exact_match_score`: The expected score to be returned when predictions are equal the gold reference. Default is 1.0.
        7. `maximum_full_mismatch_score`: The maximum score allowed to be returned when predictions are full mismatched. Default is 0.0.
        8. `full_mismatch_prediction_values`: An optional list of prediction values to use for testing full mismatches. Default is None.
            If not set, a default set of values: ["a1s", "bfsdf", "dgdfgs", "gfjgfh", "ghfjgh"]
        9. **kwargs`: Additional keyword arguments to be passed to the recipe.

    Example:
            # Test the templates with few shots
            test_card(card,num_demos=1,demo_pool_size=10)

            # Shows the step by step processing of data.
            test_card(card,debug=True)

            # In some metrics (e.g. BertScore) random predictions do not generate a score of zero so we disable this test
            test_card(card,test_full_mismatch_score_with_full_mismatch_prediction_values=False)

            # Alternatively, we can ensure the score on random predictions is less than 0.7
            test_card(card,maximum_full_mismatch_score=0.7)

            # Override the values used when running the test to check that fully mismatched values get 0 score
            test_card(card,full_mismatch_prediction_values=["NA","NONE])


    """
    if full_mismatch_prediction_values is None:
        full_mismatch_prediction_values = ["a1s", "bfsdf", "dgdfgs", "gfjgfh", "ghfjgh"]
    if settings.test_card_disable:
        logger.info(
            "test_card() functionality is disabled because unitxt.settings.test_card_disable=True or UNITXT_TEST_CARD_DISABLE environment variable is set"
        )
        return
    test_adding_to_catalog(card)
    test_metrics_exist(card)
    test_loading_from_catalog(card)

    if type(card.templates) is TemplatesDict:
        template_card_indices = card.templates.keys()
    else:
        num_templates = len(card.templates)
        template_card_indices = range(0, num_templates)
    for template_card_index in template_card_indices:
        examples = load_examples_from_standard_recipe(
            card, template_card_index=template_card_index, debug=debug, **kwargs
        )
        if test_exact_match_score_when_predictions_equal_references:
            test_correct_predictions(
                examples=examples, strict=strict, exact_match_score=exact_match_score
            )
        if test_full_mismatch_score_with_full_mismatch_prediction_values:
            test_wrong_predictions(
                examples=examples,
                strict=strict,
                maximum_full_mismatch_score=maximum_full_mismatch_score,
                full_mismatch_prediction_values=full_mismatch_prediction_values,
            )
