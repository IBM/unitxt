from unitxt import add_to_catalog
from unitxt.operators import AugmentPrefixSuffix

pattern_distribution = {
    " ": 20,
    "\\t": 10,
    "\\n": 40,
    "": 30,
}

operator = AugmentPrefixSuffix(
    augment_task_input=True,
    prefixes=pattern_distribution,
    suffixes=pattern_distribution,
    prefix_len=5,
    suffix_len=5,
    remove_existing_whitespaces=True,
)

# text = " She is riding a black horse\t\t  "
# inputs = [{"inputs": {"text": text}}]
# operator.set_task_input_fields(["text"])
# outputs = apply_operator(operator, inputs)

add_to_catalog(
    operator,
    "augmentors.augment_whitespace_prefix_and_suffix_task_input",
    overwrite=True,
)
