import os

import datasets

from .api import __file__ as _
from .artifact import __file__ as _
from .blocks import __file__ as _
from .card import __file__ as _
from .catalog import __file__ as _
from .collections import __file__ as _
from .collections_operators import __file__ as _
from .dataclass import __file__ as _
from .dataset_utils import get_dataset_artifact
from .deprecation_utils import __file__ as _
from .dialog_operators import __file__ as _
from .dict_utils import __file__ as _
from .eval_utils import __file__ as _
from .file_utils import __file__ as _
from .formats import __file__ as _
from .fusion import __file__ as _
from .generator_utils import __file__ as _
from .hf_utils import verify_versions_compatibility
from .inference import __file__ as _
from .instructions import __file__ as _
from .llm_as_judge import __file__ as _
from .loaders import __file__ as _
from .logging_utils import get_logger
from .metric import __file__ as _
from .metric_utils import __file__ as _
from .metrics import __file__ as _
from .normalizers import __file__ as _
from .operator import __file__ as _
from .operators import __file__ as _
from .parsing_utils import __file__ as _
from .processors import __file__ as _
from .random_utils import __file__ as _
from .recipe import __file__ as _
from .register import __file__ as _
from .schema import __file__ as _
from .settings_utils import get_constants
from .span_lableing_operators import __file__ as _
from .split_utils import __file__ as _
from .splitters import __file__ as _
from .standard import __file__ as _
from .stream import __file__ as _
from .stream_operators import __file__ as _
from .string_operators import __file__ as _
from .struct_data_operators import __file__ as _
from .system_prompts import __file__ as _
from .task import __file__ as _
from .templates import __file__ as _
from .text_utils import __file__ as _
from .type_utils import __file__ as _
from .utils import is_package_installed
from .validate import __file__ as _
from .version import __file__ as _
from .version import version

logger = get_logger()
constants = get_constants()


class Dataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = constants.version

    @property
    def generators(self):
        if not hasattr(self, "_generators") or self._generators is None:
            if is_package_installed("unitxt"):
                verify_versions_compatibility("dataset", self.VERSION)

                from unitxt.dataset_utils import (
                    get_dataset_artifact as get_dataset_artifact_installed,
                )

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
