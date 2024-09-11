from unitxt import add_to_catalog
from unitxt.augmentors import (
    AugmentWhitespace,
    ModelInputAugmentor,
    TaskInputsAugmentor,
)

operator = ModelInputAugmentor(operator=AugmentWhitespace())

add_to_catalog(operator, "augmentors.augment_whitespace_model_input", overwrite=True)

operator = TaskInputsAugmentor(operator=AugmentWhitespace())

add_to_catalog(operator, "augmentors.augment_whitespace_task_input", overwrite=True)
