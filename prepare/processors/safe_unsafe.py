from unitxt import add_to_catalog
from unitxt.processors import ExtractSafeUnsafeJudgment, PostProcess
from unitxt.settings_utils import get_constants

constants = get_constants()

add_to_catalog(
    PostProcess(ExtractSafeUnsafeJudgment()),
    "processors.safe_unsafe",
    overwrite=True,
)
