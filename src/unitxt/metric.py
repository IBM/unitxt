from typing import Dict, Iterable, List

import evaluate

from .api import __file__ as _
from .artifact import __file__ as _
from .blocks import __file__ as _
from .card import __file__ as _
from .catalog import __file__ as _
from .collections import __file__ as _
from .collections_operators import __file__ as _
from .dataclass import __file__ as _
from .dataset_utils import __file__ as _
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
from .logging_utils import __file__ as _
from .metric_utils import UNITXT_METRIC_SCHEMA, _compute
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

constants = get_constants()


class Metric(evaluate.Metric):
    calc_confidence_intervals: bool = True

    VERSION = constants.version

    def _info(self):
        return evaluate.MetricInfo(
            description="_DESCRIPTION",
            citation="_CITATION",
            features=UNITXT_METRIC_SCHEMA,
            codebase_urls=[constants.codebase_url],
            reference_urls=[constants.website_url],
        )

    def _compute(
        self,
        predictions: List[str],
        references: Iterable,
        flatten: bool = False,
        split_name: str = "all",
    ):
        if is_package_installed("unitxt"):
            verify_versions_compatibility("metric", self.VERSION)

            from unitxt.metric_utils import _compute as _compute_installed

            return _compute_installed(
                predictions=predictions,
                references=references,
                flatten=flatten,
                split_name=split_name,
                calc_confidence_intervals=self.calc_confidence_intervals,
            )

        return _compute(
            predictions=predictions,
            references=references,
            flatten=flatten,
            split_name=split_name,
            calc_confidence_intervals=self.calc_confidence_intervals,
        )
