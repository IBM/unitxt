from unitxt import add_to_catalog
from unitxt.blocks import ToString, ToStringStripped
from unitxt.processors import PostProcess
from unitxt.settings_utils import get_constants

constants = get_constants()

add_to_catalog(
    PostProcess(ToString()),
    "processors.to_string",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ToStringStripped()),
    "processors.to_string_stripped",
    overwrite=True,
)
