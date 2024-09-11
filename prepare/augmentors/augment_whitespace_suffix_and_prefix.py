from unitxt import add_to_catalog
from unitxt.augmentors import AugmentPrefixSuffix, TaskInputsAugmentor

pattern_distribution = {
    " ": 20,
    "\\t": 10,
    "\\n": 40,
    "": 30,
}

operator = AugmentPrefixSuffix(
    prefixes=pattern_distribution,
    suffixes=pattern_distribution,
    prefix_len=5,
    suffix_len=5,
    remove_existing_whitespaces=True,
)

augmentor = TaskInputsAugmentor(operator=operator)

# text = " She is riding a black horse\t\t  "
# inputs = [{"inputs": {"text": text}}]
# operator.set_task_input_fields(["text"])
# outputs = apply_operator(operator, inputs)

add_to_catalog(
    augmentor,
    "augmentors.augment_whitespace_prefix_and_suffix_task_input",
    overwrite=True,
)
