import os

import datasets

from .artifact import Artifact, UnitxtArtifactNotFoundError, fetch_artifact
from .artifact import __file__ as _
from .blocks import __file__ as _
from .card import __file__ as _
from .catalog import __file__ as _
from .collections import __file__ as _
from .dataclass import __file__ as _
from .dict_utils import __file__ as _
from .file_utils import __file__ as _
from .formats import __file__ as _
from .fusion import __file__ as _
from .generator_utils import __file__ as _
from .hf_utils import __file__ as _
from .instructions import __file__ as _
from .load import __file__ as _
from .loaders import __file__ as _
from .logging_utils import get_logger
from .metric import __file__ as _
from .metrics import __file__ as _
from .normalizers import __file__ as _
from .operator import __file__ as _
from .operators import __file__ as _
from .processors import __file__ as _
from .random_utils import __file__ as _
from .recipe import __file__ as _
from .register import __file__ as _
from .register import _reset_env_local_catalogs, register_all_artifacts
from .schema import __file__ as _
from .split_utils import __file__ as _
from .splitters import __file__ as _
from .standard import __file__ as _
from .stream import __file__ as _
from .task import __file__ as _
from .templates import __file__ as _
from .text_utils import __file__ as _
from .type_utils import __file__ as _
from .utils import __file__ as _
from .validate import __file__ as _
from .version import __file__ as _
from .version import version

logger = get_logger()

__default_recipe__ = "standard_recipe"


def fetch(artifact_name):
    try:
        artifact, _ = fetch_artifact(artifact_name)
        return artifact
    except UnitxtArtifactNotFoundError:
        return None


def parse(query: str):
    """Parses a query of the form 'key1=value1,key2=value2,...' into a dictionary."""
    result = {}
    kvs = query.split(",")
    if len(kvs) == 0:
        raise ValueError(
            'Illegal query: "{query}" should contain at least one assignment of the form: key1=value1,key2=value2'
        )
    for kv in kvs:
        key_val = kv.split("=")
        if (
            len(key_val) != 2
            or len(key_val[0].strip()) == 0
            or len(key_val[1].strip()) == 0
        ):
            raise ValueError(
                f'Illegal query: "{query}" with wrong assignment "{kv}" should be of the form: key=value.'
            )
        key, val = key_val
        if val.isdigit():
            result[key] = int(val)
        elif val.replace(".", "", 1).isdigit():
            result[key] = float(val)
        else:
            result[key] = val

    return result


def get_dataset_artifact(dataset_str):
    _reset_env_local_catalogs()
    register_all_artifacts()
    recipe = fetch(dataset_str)
    if recipe is None:
        args = parse(dataset_str)
        if "type" not in args:
            args["type"] = os.environ.get("UNITXT_DEFAULT_RECIPE", __default_recipe__)
        recipe = Artifact.from_dict(args)
    return recipe


class Dataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version(version)

    @property
    def generators(self):
        if not hasattr(self, "_generators") or self._generators is None:
            try:
                from unitxt.dataset import (
                    get_dataset_artifact as get_dataset_artifact_installed,
                )

                unitxt_installed = True
            except ImportError:
                unitxt_installed = False

            if unitxt_installed:
                logger.info("Loading with installed unitxt library...")
                dataset = get_dataset_artifact_installed(self.config.name)
            else:
                logger.info("Loading with huggingface unitxt copy...")
                dataset = get_dataset_artifact(self.config.name)

            self._generators = dataset()

        return self._generators

    def _info(self):
        return datasets.DatasetInfo()

    def _split_generators(self, _):
        return [
            datasets.SplitGenerator(name=name, gen_kwargs={"split_name": name})
            for name in self.generators.keys()
        ]

    def _generate_examples(self, split_name):
        generator = self.generators[split_name]
        yield from enumerate(generator)

    def _download_and_prepare(
        self, dl_manager, verification_mode, **prepare_splits_kwargs
    ):
        return super()._download_and_prepare(
            dl_manager, "no_checks", **prepare_splits_kwargs
        )
