import os.path
import tempfile

from .. import add_to_catalog
from ..artifact import fetch_artifact

TEMP_NAME = 'tmp_name'


def test_adding_to_catalog(card):
    with tempfile.TemporaryDirectory() as tmp_dir:
        add_to_catalog(card, TEMP_NAME, overwrite=True, catalog_path=tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, TEMP_NAME + '.json')), 'Card was not added to catalog'

def test_metrics_exist(card):
    for metric_name in card.task.metrics:
        metric, _ = fetch_artifact(metric_name)

def test_loading_to_catalog():
    pass


def test_with_common_recipe():
    pass


def test_with_eval():
    pass


def test_card(card):
    test_adding_to_catalog(card)
    test_metrics_exist(card)
    test_loading_to_catalog()
    test_with_common_recipe()
    test_with_eval()
