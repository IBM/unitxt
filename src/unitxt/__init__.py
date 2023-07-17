from .register import register_all_artifacts, register_local_catalog
from .load import load_dataset
from .catalog import add_to_catalog

register_all_artifacts()

dataset_url = "unitxt/data"
metric_url = "unitxt/metric"
