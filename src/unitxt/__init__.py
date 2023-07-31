import os
import random

from .catalog import add_to_catalog
from .load import load_dataset
from .register import register_all_artifacts, register_local_catalog

register_all_artifacts()
random.seed(0)
dataset_file = os.path.join(os.path.dirname(__file__), "dataset.py")
metric_file = os.path.join(os.path.dirname(__file__), "metric.py")

dataset_url = "unitxt/data"
metric_url = "unitxt/metric"
