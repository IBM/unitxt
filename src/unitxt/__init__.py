import os
import random

from . import logging_utils
from .api import evaluate, load, load_dataset
from .catalog import add_to_catalog
from .register import register_all_artifacts, register_local_catalog
from .settings_utils import Settings
from .version import version

__version__ = version

register_all_artifacts()
random.seed(0)
dataset_file = os.path.join(os.path.dirname(__file__), "dataset.py")
metric_file = os.path.join(os.path.dirname(__file__), "metric.py")
local_catalog_path = os.path.join(os.path.dirname(__file__), "catalog")

dataset_url = "unitxt/data"
metric_url = "unitxt/metric"

settings = Settings()
settings.allow_unverified_code = False
