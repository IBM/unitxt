from unitxt import add_to_catalog
from unitxt.blocks import ToString, ToStringStripped
from unitxt.processors import PostProcess

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
