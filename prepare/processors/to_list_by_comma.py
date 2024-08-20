from unitxt import add_to_catalog
from unitxt.processors import PostProcess, ToListByComma, ToListByCommaSpace
from unitxt.settings_utils import get_constants

constants = get_constants()
add_to_catalog(
    PostProcess(ToListByComma()),
    "processors.to_list_by_comma",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ToListByComma(), process_prediction=False),
    "processors.to_list_by_comma_from_references",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ToListByCommaSpace()),
    "processors.to_list_by_comma_space",
    overwrite=True,
)
