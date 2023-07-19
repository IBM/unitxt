from .register import register_all_artifacts, register_local_catalog
from .load import load_dataset
from .catalog import add_to_catalog
import os

register_all_artifacts()

dataset_file = os.path.join(os.path.dirname(__file__), "dataset.py")
metric_file = os.path.join(os.path.dirname(__file__), "metric.py")

dataset_url = "unitxt/data"
metric_url = "unitxt/metric"
