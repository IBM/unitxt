import json
import logging
import math
import os.path
import tempfile

from .. import add_to_catalog, register_local_catalog
from ..artifact import fetch_artifact
from ..metric import _compute
from ..standard import StandardRecipe
from ..text_utils import print_dict

TEMP_NAME = "tmp_name"


def test_adding_to_catalog(card):
    with tempfile.TemporaryDirectory() as tmp_dir:
        add_to_catalog(card, TEMP_NAME, overwrite=True, catalog_path=tmp_dir, verbose=False)
        assert os.path.exists(os.path.join(tmp_dir, TEMP_NAME + ".json")), "Card was not added to catalog"


def test_metrics_exist(card):
    for metric_name in card.task.metrics:
        metric, _ = fetch_artifact(metric_name)


def test_loading_from_catalog(card):
    with tempfile.TemporaryDirectory() as tmp_dir:
        add_to_catalog(card, TEMP_NAME, overwrite=True, catalog_path=tmp_dir, verbose=False)
        register_local_catalog(tmp_dir)
        card_, _ = fetch_artifact(TEMP_NAME)
        assert json.dumps(card_.to_dict(), sort_keys=True) == json.dumps(
            card.to_dict(), sort_keys=True
        ), "Card loaded is not equal to card stored"


def load_examples_from_standard_recipe(card, tested_split, template_card_index=None):
    if template_card_index == None:
        if card.templates:
            try:  # named templates (dict)
                template_item = next(iter(card.templates.keys()))
            except AttributeError:  # template list
                template_item = 0

    num_instructions = len(card.instructions) if card.instructions else 0
    recipe = StandardRecipe(
        card=card,
        demos_pool_size=100,
        demos_taken_from=tested_split,
        num_demos=3,
        template_card_index=template_card_index,
        instruction_card_index=0 if num_instructions else None,
        loader_limit=200,
    )
    multi_stream = recipe()
    stream = multi_stream[tested_split]
    try:
        examples = list(stream.take(3))
    except ValueError as e:
        raise ValueError(
            "Try setting streaming=False in LoadHF in your card. For example: LoadHF(path='glue', name='mrpc', streaming=False). Org error message:",
            e,
        )
    print("3 Examples: ")
    for example in examples:
        print_dict(example)
        print("\n")

    return examples


def test_with_eval(card, tested_split, strict=True, exact_match_score=1.0, full_mismatch_score=0.0):
    num_templates = len(card.templates)
    for template_item in range(0, num_templates):
        examples = load_examples_from_standard_recipe(card, tested_split, template_item)
    # metric = evaluate.load('unitxt/metric')
    predictions = []
    for example in examples:
        predictions.append(example["references"][0] if len(example["references"]) > 0 else [])

    results = _compute(predictions=predictions, references=examples)
    if not exact_match_score == None and not math.isclose(
        results[0]["score"]["global"]["groups_mean_score"], exact_match_score
    ):
        message = (
            f"Metric on examples equal predicions is no {exact_match_score}, but {results[0]['score']['global']['groups_mean_score']}."
            f" If you are using matthews_correlation, it is possible that this is because all your examples come from "
            f"one class. Consider setting strict=False"
        )
        if strict:
            raise AssertionError(message)
        else:
            print(f"Warning: {message}")

    predictions = ["a1s", "bfsdf", "dgdfgs", "gfjgfh", "ghfjgh"]
    results = _compute(predictions=predictions, references=examples)
    if not full_mismatch_score == None and results[0]["score"]["global"]["groups_mean_score"] != full_mismatch_score:
        print(
            f"Warning: metric on random predictions is not {full_mismatch_score}, but {results[0]['score']['global']['groups_mean_score']} "
        )


def test_card(card, tested_split="train", strict=True, exact_match_score=1.0, full_mismatch_score=0.0):
    test_adding_to_catalog(card)
    test_metrics_exist(card)
    test_loading_from_catalog(card)
    test_with_eval(
        card, tested_split, strict=strict, exact_match_score=exact_match_score, full_mismatch_score=full_mismatch_score
    )
