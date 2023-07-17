import json
import logging
import os.path
import tempfile

from .. import add_to_catalog
from ..artifact import fetch_artifact
from .. import register_local_catalog
from ..common import CommonRecipe
from ..metric import _compute
from ..text_utils import print_dict

TEMP_NAME = 'tmp_name'


def test_adding_to_catalog(card):
    with tempfile.TemporaryDirectory() as tmp_dir:
        add_to_catalog(card, TEMP_NAME, overwrite=True, catalog_path=tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, TEMP_NAME + '.json')), 'Card was not added to catalog'


def test_metrics_exist(card):
    for metric_name in card.task.metrics:
        metric, _ = fetch_artifact(metric_name)


def test_loading_from_catalog(card):
    with tempfile.TemporaryDirectory() as tmp_dir:
        add_to_catalog(card, TEMP_NAME, overwrite=True, catalog_path=tmp_dir)
        register_local_catalog(tmp_dir)
        card_, _ = fetch_artifact(TEMP_NAME)
        assert json.dumps(card_.to_dict(), sort_keys=True) == json.dumps(card.to_dict(), sort_keys=True), \
            'Card loaded is not equal to card stored'


def load_examples_from_common_recipe(card):
    num_templates = len(card.templates) if card.templates else 0
    num_instructions = len(card.instructions) if card.instructions else 0
    recipe = CommonRecipe(
        card=card,
        demos_pool_size=100,
        num_demos=3,
        template_item=0 if num_templates else None,
        instruction_item=0 if num_instructions else None
    )
    multi_stream = recipe()
    stream = next(iter(multi_stream.values()))
    examples = list(stream.take(5))
    print('5 Examples: ')
    for example in examples:
        print_dict(example)
        print('\n')

    return examples


def test_with_eval(card):
    examples = load_examples_from_common_recipe(card)
    #metric = evaluate.load('unitxt/metric')
    predictions = []
    for example in examples:
        predictions.append(example['references'][0] if len(example['references']) > 0 else [])

    results = _compute(predictions=predictions, references=examples)
    assert results[0]['score']['global']['groups_mean_score'] == 1.0, \
        f"Metric on examples equal predicions is no 1.0, but {results[0]['score']['global']['groups_mean_score']}"
    predictions = ['a1s', 'bfsdf', 'dgdfgs', 'gfjgfh', 'ghfjgh']
    results = _compute(predictions=predictions, references=examples)
    if results[0]['score']['global']['groups_mean_score'] != 0.0:
        print(f"Warning: metric on rundom predicions is different than zero "
              f"({results[0]['score']['global']['groups_mean_score']})")



def test_card(card):
    test_adding_to_catalog(card)
    test_metrics_exist(card)
    test_loading_from_catalog(card)
    test_with_eval(card)
