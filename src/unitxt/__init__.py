import random
import warnings

from .api import evaluate, infer, load, load_dataset, post_process, produce
from .catalog import add_to_catalog, get_from_catalog
from .logging_utils import get_logger
from .register import register_all_artifacts, register_local_catalog
from .settings_utils import get_constants, get_settings

register_all_artifacts()
random.seed(0)

constants = get_constants()
settings = get_settings()
logger = get_logger()

__version__ = constants.version

dataset_file = constants.dataset_file
metric_file = constants.metric_file
local_catalog_path = constants.local_catalog_path

dataset_url = constants.dataset_url
metric_url = constants.metric_url
