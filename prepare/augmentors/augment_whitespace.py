from src.unitxt import add_to_catalog
from src.unitxt.operators import AugmentWhitespace

operator = AugmentWhitespace(augment_model_input=True)

add_to_catalog(operator, "augmentors.augment_whitespace_model_input", overwrite=True)

operator = AugmentWhitespace(augment_task_input=True)

add_to_catalog(operator, "augmentors.augment_whitespace_task_input", overwrite=True)
